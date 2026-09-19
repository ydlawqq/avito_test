"""Формирование датасета для обучения XGBoost-реранкера поверх BM25 с бустами.

Для каждой сессии (запроса) воспроизводится ПРОДАКШН-вход реранкера:
    чистый BM25 -> сырой пул top-5000 (cfg["bm25"]["boost_pool"])
                -> буст локации/категории score * (1 + boost * match)
                -> топ-`--depth` (1000) переранжированных объявлений.

Каждая строка датасета — пара (запрос, объявление) из этого топа:
    label = 1, если объявление пользователь выбирал по этому запросу
            (relevant_items сессии), иначе 0.
Признаки — build_features() из solution/src/rerank/xgb_features.py (общие
с инференсом в scripts/make_answer.py --method xgb).

Сторона запроса: ТОЛЬКО search_* признаки (search_query, search_location_id,
search_is_delivery_search, search_infm_params_text, search_category) — как
в benchmark_queries.parquet. Сторона объявления — из корпуса (benchmark_items).

Сессии-источники:
    * по умолчанию artifacts/validation.parquet (2452 сессии из train.parquet);
    * --train-from raw_data/train.parquet: ВСЕ ~26.5k сессий train.parquet
      (build_validation с size=10^9) — значительно больше данных на обучение.

Сплит по сессиям в детерминированном md5-порядке (как в validation.parquet):
первые `--train-frac` — split='train', остальные — split='eval'. Сессии eval
не пересекаются с validation.parquet (та целиком лежит в первых 70% порядка),
поэтому метрики на eval честны и для полного прогона.

Опционально --neg-frac < 1: в TRAIN-сплите сэмплируются негативы с
bm25_rank > 50 (голова пула сохраняется целиком) — ускоряет обучение без
изменения распределения верхней части пула. EVAL не трогается никогда.

Запуск из корня проекта (~10-15 мин CPU на 2452 сессии, ~2 ч на весь train):
    python experiments/xgb_rerank/build_dataset.py                 # 2452 сессии
    python experiments/xgb_rerank/build_dataset.py --size 5        # smoke
    python experiments/xgb_rerank/build_dataset.py \
        --train-from raw_data/train.parquet                        # все сессии
    python experiments/xgb_rerank/build_dataset.py --neg-frac 0.3 \
        --train-from raw_data/train.parquet                        # быстрее
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# experiments/xgb_rerank -> корень проекта
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "solution"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from recall_at_k_bm25 import QueryLemmatizer, check_parquet
from src.evaluation.validation import build_validation
from src.pipeline.common import artifacts_dir, build_bm25, load_config
from src.rerank.xgb_features import (
    NUMERIC_FEATURES,
    ItemSideIndex,
    boosted_pool,
    build_features,
    prepare_item_side,
)

# голова пула, которая в TRAIN-сплите сохраняется целиком при --neg-frac < 1
NEG_HEAD = 50
BATCH_SESSIONS = 64  # размер буфера записи parquet


def dataset_schema() -> pa.Schema:
    fields = [
        ("query_id", pa.string()),
        ("item_id", pa.string()),
        ("label", pa.int8()),
        ("split", pa.string()),
    ]
    fields += [(name, pa.float32()) for name in NUMERIC_FEATURES]
    fields += [("microcat_id", pa.int64()), ("search_loc", pa.int64())]
    return pa.schema(fields)


def session_frame(
    qrow,
    corpus_idx: np.ndarray,
    raw_scores: np.ndarray,
    boosted_scores: np.ndarray,
    raw_ranks: np.ndarray,
    labels: np.ndarray,
    split: str,
    qtokens: list[str],
    item_index: ItemSideIndex,
) -> pd.DataFrame:
    X_num, micro, search_loc = build_features(
        qrow, corpus_idx, raw_scores, boosted_scores, raw_ranks, qtokens, item_index
    )
    df = pd.DataFrame(X_num, columns=NUMERIC_FEATURES)
    df.insert(0, "split", split)
    df.insert(0, "label", labels.astype(np.int8))
    df.insert(0, "item_id", item_index.doc_ids[corpus_idx])
    df.insert(0, "query_id", qrow.query_id)
    df["microcat_id"] = micro
    df["search_loc"] = search_loc
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "config.yaml"))
    parser.add_argument("--validation", default=None,
                        help="validation.parquet (по умолчанию artifacts/validation.parquet)")
    parser.add_argument("--train-from", default=None,
                        help="raw train.parquet: сессии ВСЕГО трейна вместо валидации")
    parser.add_argument("--size", type=int, default=None, help="ограничить число сессий")
    parser.add_argument("--depth", type=int, default=None,
                        help="размер переранжированного пула (по умолчанию xgb_rerank.depth=1000)")
    parser.add_argument("--pool", type=int, default=None,
                        help="сырой BM25-пул до бустов (по умолчанию bm25.boost_pool)")
    parser.add_argument("--loc-boost", type=float, default=None)
    parser.add_argument("--cat-boost", type=float, default=None)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--neg-frac", type=float, default=1.0,
                        help="доля сохраняемых негативов с rank>50 в TRAIN-сплите")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out",
                        default=str(PROJECT_ROOT / "artifacts" / "xgb_rerank" / "dataset.parquet"))
    args = parser.parse_args()

    t0 = time.time()
    cfg = load_config(args.config)
    b = cfg["bm25"]
    depth = args.depth or int(cfg.get("xgb_rerank", {}).get("depth", 1000))
    pool = args.pool or int(b.get("boost_pool", 5000))
    loc_boost = (args.loc_boost if args.loc_boost is not None
                 else float(b.get("location_boost", 2.0)))
    cat_boost = (args.cat_boost if args.cat_boost is not None
                 else float(b.get("category_boost", 0.0)))
    print(f"depth={depth}, pool={pool}, loc_boost={loc_boost}, cat_boost={cat_boost}")

    # ------------------------------------------------------------- сессии
    artifacts = artifacts_dir(cfg)
    if args.train_from:
        sessions = build_validation(
            args.train_from, artifacts / "items_processed.parquet", size=10**9
        )
        source = "train.parquet (все сессии)"
    else:
        val_path = (Path(args.validation) if args.validation
                    else artifacts / "validation.parquet")
        check_parquet(val_path, "validation.parquet")
        sessions = pd.read_parquet(val_path)
        source = str(val_path)
    if args.size:
        sessions = sessions.head(args.size)
    n_train = int(len(sessions) * args.train_frac)
    print(f"Сессии: {len(sessions)} из {source} | "
          f"train={n_train}, eval={len(sessions) - n_train}")

    relevance = {r.query_id: set(r.relevant_items) for r in sessions.itertuples()}

    # ------------------------------------------------------------- корпус
    check_parquet(artifacts / "items_processed.parquet", "items_processed.parquet")
    items = prepare_item_side(cfg)
    item_index = ItemSideIndex(items)
    retriever = build_bm25(items, b)
    print(f"Корпус: {len(items)} объявлений ({time.time() - t0:.0f} c)")


    # ------------------------------------------------------ генерация строк
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(out_path, dataset_schema(), compression="zstd")

    lemmatizer = QueryLemmatizer()
    rng = np.random.default_rng(args.seed)
    buf: list[pd.DataFrame] = []
    stats = {
        split: {"rows": 0, "pos": 0, "sessions": 0, "no_pos": 0}
        for split in ("train", "eval")
    }
    rank_buckets = [50, 100, 250, 500, 1000]
    pos_rank_hist = {k: 0 for k in rank_buckets}
    pos_total = 0
    pool_sizes: list[int] = []

    def flush() -> None:
        nonlocal buf
        if buf:
            writer.write_table(pa.Table.from_pandas(
                pd.concat(buf, ignore_index=True), schema=dataset_schema(),
                preserve_index=False,
            ))
            buf = []

    for si, qrow in enumerate(tqdm(sessions.itertuples(), total=len(sessions),
                                   desc="Датасет")):
        split = "train" if si < n_train else "eval"
        tokens = lemmatizer(qrow.search_query, qrow.search_infm_params_text)
        corpus_idx, raw_s, boost_s, raw_r = boosted_pool(
            retriever, tokens, item_index,
            qrow.search_location_id, qrow.search_category,
            depth=depth, pool=pool, loc_boost=loc_boost, cat_boost=cat_boost,
        )
        if len(corpus_idx) == 0:
            continue
        rel = relevance.get(qrow.query_id, set())
        labels = np.fromiter(
            (1 if iid in rel else 0 for iid in item_index.doc_ids[corpus_idx]),
            dtype=np.int8, count=len(corpus_idx),
        )

        if split == "train" and args.neg_frac < 1.0:
            # сохраняем: все позитивы + голову пула (rank<=50) + часть хвоста
            keep = (labels > 0) | (np.arange(len(labels)) < NEG_HEAD)
            tail = np.flatnonzero(~keep)
            n_take = int(len(tail) * args.neg_frac)
            if n_take < len(tail) and n_take > 0:
                drop = rng.choice(tail, size=len(tail) - n_take, replace=False)
                keep[drop] = False
            sel = np.flatnonzero(keep)
            corpus_idx, raw_s, boost_s, raw_r, labels = (
                corpus_idx[sel], raw_s[sel], boost_s[sel], raw_r[sel], labels[sel],
            )

        st = stats[split]
        st["rows"] += len(labels)
        st["pos"] += int(labels.sum())
        st["sessions"] += 1
        st["no_pos"] += int(labels.sum() == 0)
        pool_sizes.append(len(labels))

        ranks = np.arange(len(labels))
        rel_pos = ranks[labels > 0]
        pos_total += len(rel_pos)
        for k in rank_buckets:
            pos_rank_hist[k] += int((rel_pos < k).sum())

        buf.append(session_frame(
            qrow, corpus_idx, raw_s, boost_s, raw_r, labels, split, tokens, item_index,
        ))
        if len(buf) >= BATCH_SESSIONS:
            flush()
    flush()
    writer.close()

    # ------------------------------------------------------------ статистика
    rows_total = sum(s["rows"] for s in stats.values())
    print(f"\nДатасет сохранён: {out_path} ({rows_total:,} строк, {time.time() - t0:.0f} c)")
    for split, st in stats.items():
        share = st["pos"] / st["rows"] if st["rows"] else 0.0
        print(f"  {split:>5}: сессий={st['sessions']}, строк={st['rows']:,}, "
              f"позитивов={st['pos']:,} ({share:.3%}), "
              f"сессий без позитивов в пуле={st['no_pos']}")
    if pos_total:
        print("Позитивы по рангу в переранжированном пуле (headroom реранкера):")
        prev = 0
        for k in rank_buckets:
            print(f"  rank <= {k:>5}: {pos_rank_hist[k] / pos_total:.4f} "
                  f"(+{pos_rank_hist[k] - prev} шт.)")
            prev = pos_rank_hist[k]
    print(f"Средний размер пула: {np.mean(pool_sizes):.0f}, "
          f"медиана: {np.median(pool_sizes):.0f}")


if __name__ == "__main__":
    main()
