"""Этап A эксперимента «BM25 → пул 10k → dense»: пулы чистого BM25.

Для каждой сессии локальной валидации берёт top-`--pool` объявлений по ЧИСТОМУ
BM25 (параметры из cfg["bm25"], бусты локации/категории НЕ применяются —
контрольная точка recall@10k = 0.9668 на полном корпусе 189k, см.
experiments/README.md) и сохраняет пул (query_id, item_id, bm25_score)
в parquet. Это вход для experiments/dense_rerank_10k.py, который кодирует
только объединение пулов — вместо индексации всего корпуса.

Запуск (CPU) из корня проекта:
    python experiments/bm25_pool.py
    python experiments/bm25_pool.py --size 100 --pool 10000
    python experiments/bm25_pool.py --out artifacts/bm25_pool.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "solution"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # recall_at_k_bm25

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from recall_at_k_bm25 import QueryLemmatizer, check_parquet
from src.evaluation.validation import load_validation, validation_relevance
from src.metrics.recall import recall_at_k
from src.pipeline.common import artifacts_dir, build_bm25, load_config, load_items

# контрольные k для печатаемой таблицы (< pool); сам pool тоже выводится
KS: tuple[int, ...] = (50, 250, 500, 1000)


def pool_idx_scores(retriever, tokens: list[str], pool: int):
    """Индексы и скоры top-`pool` документов по чистому BM25 (score > 0, по убыванию)."""
    scores = retriever.score_tokens(tokens)
    n = len(scores)
    take = min(int(pool), n)
    if take <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    idx = np.argpartition(-scores, take - 1)[:take]
    idx = idx[scores[idx] > 0]
    idx = idx[np.argsort(-scores[idx], kind="stable")]
    return idx, scores[idx].astype("float32")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "config.yaml"))
    parser.add_argument("--size", type=int, default=None,
                        help="сколько сессий валидации брать (по умолчанию все 2452)")
    parser.add_argument("--pool", type=int, default=10000,
                        help="размер BM25-пула на запрос (по умолчанию 10000)")
    parser.add_argument("--out", default=None,
                        help="куда писать пул (по умолчанию artifacts/bm25_pool.parquet)")
    parser.add_argument("--validation", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    artifacts = artifacts_dir(cfg)
    val_path = Path(args.validation) if args.validation else artifacts / "validation.parquet"
    items_path = artifacts / "items_processed.parquet"
    out_path = Path(args.out) if args.out else artifacts / "bm25_pool.parquet"

    check_parquet(val_path, "validation.parquet")
    check_parquet(items_path, "items_processed.parquet")

    validation = load_validation(val_path)
    if args.size:
        validation = validation.head(args.size)
    print(f"Сессии валидации: {len(validation)} | пул на запрос: {args.pool}")

    relevance = validation_relevance(validation)
    items = load_items(cfg)
    print(f"Корпус: {len(items)} объявлений")
    retriever = build_bm25(items, cfg["bm25"])
    doc_ids = retriever.doc_ids
    b = cfg["bm25"]
    print(f"Чистый BM25: title_weight={b.get('title_weight')}, "
          f"use_params={b.get('use_params')}, use_description={b.get('use_description')}")

    query_tokens = QueryLemmatizer()
    schema = pa.schema([
        ("query_id", pa.string()),
        ("item_id", pa.string()),
        ("bm25_score", pa.float32()),
    ])
    writer = pq.ParquetWriter(out_path, schema, compression="zstd")

    ks = sorted(set([k for k in KS if k <= args.pool] + [args.pool]))
    recall_sum = {k: 0.0 for k in ks}
    n = 0
    try:
        for row in tqdm(validation.itertuples(), total=len(validation),
                        desc=f"BM25 top-{args.pool}"):
            rel = relevance[row.query_id]
            tokens = query_tokens(row.search_query, row.search_infm_params_text)
            idx, scores = pool_idx_scores(retriever, tokens, args.pool)
            ids = [doc_ids[int(i)] for i in idx]
            writer.write_table(pa.table({
                "query_id": pa.array([row.query_id] * len(ids), pa.string()),
                "item_id": pa.array(ids, pa.string()),
                "bm25_score": pa.array(scores, pa.float32()),
            }, schema=schema))
            for k in ks:
                recall_sum[k] += recall_at_k(ids, rel, k=k)
            n += 1
    finally:
        writer.close()

    print(f"\nПул сохранён: {out_path} ({n} сессий)")
    print(f"\n== Контроль: Recall@k чистого BM25 (порядок пула), {n} сессий ==")
    print(f"{'k':>8s} {'Recall@k':>10s}")
    for k in ks:
        print(f"{k:>8d} {(recall_sum[k] / n if n else 0.0):>10.4f}")


if __name__ == "__main__":
    main()
