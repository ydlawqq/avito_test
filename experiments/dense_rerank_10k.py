"""Этап B (GPU): dense-ранжирование и hybrid внутри BM25-пула 10k.

Читает пул этапа A (artifacts/bm25_pool.parquet, чистый BM25 top-10k на запрос),
кодирует объединение всех item_id из пулов (НЕ весь корпус) с возобновляемым
кэшем float16 по шардам, затем для каждого запроса ищет его только среди чанков
ЕГО пула (локальный faiss IndexFlatIP) и печатает Recall@k для вариантов:

  bm25            порядок пула (контроль этапа A);
  bm25+loc        скор пула с бустом локации score*(1+loc_boost);
  dense           косинус: max по чанкам item'а внутри пула;
  rrf(w=..)       RRF(bm25, dense), вес dense = w из сетки;
  rrf+loc(w=..)   RRF(bm25+loc, dense).

Запуск (GPU-машина) из корня проекта — нужны artifacts/{validation,
items_processed,bm25_pool}.parquet:
    python experiments/dense_rerank_10k.py --size 100
    python experiments/dense_rerank_10k.py                    # все сессии пула
    python experiments/dense_rerank_10k.py --dense-weights 0.5,1,2,4

Кэш эмбеддингов: artifacts/dense/pool_shards/shard_XXXX.npy (float16,
порядок чанков детерминирован — прерванный прогон продолжается с места остановки).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "solution"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))       # build_dense_index.build_chunks
sys.path.insert(0, str(Path(__file__).resolve().parent))  # recall_at_k_bm25

import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm

from build_dense_index import build_chunks
from recall_at_k_bm25 import check_parquet
from src.evaluation.validation import load_validation, validation_relevance
from src.metrics.recall import recall_at_k
from src.pipeline.common import artifacts_dir, load_config, load_items

SHARD_ROWS = 200_000  # строк float16 в одном шарде кэша


def _parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.split(r"[,\s]+", text.strip()) if x)


def _parse_floats(text: str) -> tuple[float, ...]:
    return tuple(float(x) for x in re.split(r"[,\s]+", text.strip()) if x)


def encode_chunks(chunk_texts: list[str], shard_dir: Path, model, batch_size: int) -> np.ndarray:
    """Эмбеддинги всех чанков (float16, L2-нормализованные) с кэшем по шардам."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    done = [np.load(p) for p in
            sorted(shard_dir.glob("shard_*.npy"), key=lambda p: p.stem.split("_")[1])]
    offset = sum(a.shape[0] for a in done)
    total = len(chunk_texts)
    print(f"Чанков к кодированию: {total} | уже в кэше: {offset}")

    if offset > total:
        raise RuntimeError(f"Кэш ({offset}) больше числа чанков ({total}) — "
                           f"удалите {shard_dir} и повторите")
    if offset < total:
        rest = chunk_texts[offset:]
        buf: list[np.ndarray] = []
        shard_no = len(done)
        n_batches = (len(rest) + batch_size - 1) // batch_size
        for start in tqdm(range(0, len(rest), batch_size), total=n_batches,
                          desc="Эмбеддинги"):
            emb = model.encode(rest[start:start + batch_size], batch_size=batch_size,
                               normalize_embeddings=True, show_progress_bar=False)
            buf.append(np.asarray(emb, dtype="float16"))
            if sum(b.shape[0] for b in buf) >= SHARD_ROWS:
                np.save(shard_dir / f"shard_{shard_no:04d}.npy", np.concatenate(buf))
                shard_no += 1
                buf = []
        if buf:
            np.save(shard_dir / f"shard_{shard_no:04d}.npy", np.concatenate(buf))
            buf = []

    shards = [np.load(p) for p in
              sorted(shard_dir.glob("shard_*.npy"), key=lambda p: p.stem.split("_")[1])]
    emb = np.concatenate(shards)
    assert emb.shape[0] == total, f"кэш {emb.shape[0]} != чанков {total}"
    print(f"Эмбеддинги готовы: {emb.shape}")
    return emb


def rank_positions(order: np.ndarray) -> np.ndarray:
    """order — перестановка индексов best-first; вернуть rank[doc] (1..N)."""
    rank = np.empty(len(order), dtype=np.float64)
    rank[order] = np.arange(1, len(order) + 1)
    return rank


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "config.yaml"))
    parser.add_argument("--size", type=int, default=None,
                        help="сколько сессий оценивать (по умолчанию все в пуле)")
    parser.add_argument("--pool-file", default=None,
                        help="пул этапа A (по умолчанию artifacts/bm25_pool.parquet)")
    parser.add_argument("--ks", default="50,250,500,1000")
    parser.add_argument("--dense-weights", default="0.5,1,2,4")
    parser.add_argument("--loc-boost", type=float, default=None,
                        help="буст локации для bm25+loc (по умолчанию из конфига)")
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--model", default=None, help="переопределить dense.model_name")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None,
                        help="CSV с результатами (по умолчанию artifacts/dense_rerank_10k.csv)")
    args = parser.parse_args()

    ks = _parse_ints(args.ks)
    weights = _parse_floats(args.dense_weights)

    cfg = load_config(args.config)
    dcfg = cfg["dense"]
    artifacts = artifacts_dir(cfg)
    val_path = artifacts / "validation.parquet"
    items_path = artifacts / "items_processed.parquet"
    pool_path = Path(args.pool_file) if args.pool_file else artifacts / "bm25_pool.parquet"
    check_parquet(val_path, "validation.parquet")
    check_parquet(items_path, "items_processed.parquet")
    check_parquet(pool_path, "bm25_pool.parquet")

    validation = load_validation(val_path)
    pool = pd.read_parquet(pool_path)
    if args.size:
        validation = validation.head(args.size)
        pool = pool[pool["query_id"].isin(set(validation["query_id"]))]
    relevance = validation_relevance(validation)
    print(f"Сессии: {len(validation)} | строк пула: {len(pool)}")

    pool_groups = {qid: (g["item_id"].to_numpy(), g["bm25_score"].to_numpy())
                   for qid, g in pool.groupby("query_id", sort=False)}

    items = load_items(cfg)
    needed = set(pool["item_id"].unique())
    items = items[items["item_id"].isin(needed)]
    print(f"Объединение пулов: {len(items)} объявлений "
          f"({len(items) / 189_212:.0%} корпуса)")

    chunk_texts: list[str] = []
    item_chunk_rows: dict[str, np.ndarray] = {}
    for row in tqdm(items.itertuples(), total=len(items), desc="Чанкование"):
        chunks = build_chunks(row.title_raw, row.params_raw, row.desc_raw,
                              chunk_words=dcfg["chunk_words"],
                              chunk_overlap=dcfg["chunk_overlap"],
                              max_chunks=dcfg["max_chunks_per_item"])
        rows = np.arange(len(chunk_texts), len(chunk_texts) + len(chunks))
        chunk_texts.extend(chunks)
        item_chunk_rows[row.item_id] = rows

    from sentence_transformers import SentenceTransformer
    model_name = args.model or dcfg["model_name"]
    batch_size = args.batch_size or dcfg.get("batch_size", 256)
    print(f"Модель: {model_name} | batch: {batch_size}")
    model = SentenceTransformer(model_name, device=args.device)
    if str(model.device).startswith("cuda"):
        model.half()  # fp16 на GPU: ~x2 быстрее, качество почти не теряет

    emb = encode_chunks(chunk_texts, artifacts / "dense" / "pool_shards",
                        model, batch_size)

    loc_map = dict(zip(items["item_id"], items["item_location_id"]))
    loc_boost = (args.loc_boost if args.loc_boost is not None
                 else cfg["bm25"].get("location_boost", 2.0))
    rrf_k = args.rrf_k

    variants = (["bm25", "bm25+loc", "dense"]
                + [f"rrf(w={w:g})" for w in weights]
                + [f"rrf+loc(w={w:g})" for w in weights])
    acc = {v: {k: 0.0 for k in ks} for v in variants}
    n = 0

    for row in tqdm(validation.itertuples(), total=len(validation), desc="Оценка"):
        rel = relevance[row.query_id]
        group = pool_groups.get(row.query_id)
        if group is None or len(group[0]) == 0:
            n += 1
            continue
        ids, bm25_scores = group
        ids = list(ids)
        m = len(ids)

        # -- dense: полный порядок пула по max-скору чанков item'а
        parts = [item_chunk_rows[i] for i in ids]
        chunk_rows = np.concatenate(parts)
        vecs = emb[chunk_rows].astype("float32")
        chunk_item = np.repeat(np.arange(m), [len(p) for p in parts])
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        q_text = " ".join(x for x in [str(row.search_query),
                                       "" if row.search_infm_params_text is None
                                       else str(row.search_infm_params_text)]
                           if x and x != "nan")
        qv = model.encode([q_text], normalize_embeddings=True)[0].astype("float32")
        D, I = index.search(qv.reshape(1, -1), len(vecs))
        best = np.full(m, -np.inf)
        np.maximum.at(best, chunk_item[I[0]], D[0])
        dense_order = np.argsort(-best, kind="stable")

        # -- bm25: пул уже отсортирован по убыванию скора (этап A)
        bm25_order = np.arange(m)
        loc = np.array([loc_map.get(i, -1) for i in ids])
        q_loc = row.search_location_id
        match = (loc == q_loc) if q_loc is not None and not pd.isna(q_loc) \
            else np.zeros(m, dtype=bool)
        boosted = bm25_scores * (1.0 + loc_boost * match)
        boost_order = np.argsort(-boosted, kind="stable")

        r_bm25 = rank_positions(bm25_order)
        r_boost = rank_positions(boost_order)
        r_dense = rank_positions(dense_order)

        rankings = {
            "bm25": bm25_order,
            "bm25+loc": boost_order,
            "dense": dense_order,
        }
        for w in weights:
            fused = 1.0 / (rrf_k + r_bm25) + w / (rrf_k + r_dense)
            rankings[f"rrf(w={w:g})"] = np.argsort(-fused, kind="stable")
            fused_loc = 1.0 / (rrf_k + r_boost) + w / (rrf_k + r_dense)
            rankings[f"rrf+loc(w={w:g})"] = np.argsort(-fused_loc, kind="stable")

        for v, order in rankings.items():
            ranked = [ids[int(i)] for i in order]
            for k in ks:
                acc[v][k] += recall_at_k(ranked, rel, k=k)
        n += 1

    print(f"\n== Dense/hybrid внутри BM25-пула, {n} сессий ==")
    print(f"{'variant':>16s}" + "".join(f"{('R@' + str(k)):>10s}" for k in ks))
    rows = []
    for v in variants:
        vals = [acc[v][k] / n if n else 0.0 for k in ks]
        print(f"{v:>16s}" + "".join(f"{x:>10.4f}" for x in vals))
        rows.extend({"variant": v, "k": k, "recall": val}
                    for k, val in zip(ks, vals))

    out_path = Path(args.out) if args.out else artifacts / "dense_rerank_10k.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nСохранено: {out_path}")


if __name__ == "__main__":
    main()
