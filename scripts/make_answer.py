"""Формирование ответа answer.csv для benchmark-запросов.

Запуск:
    python scripts/make_answer.py --method bm25    # CPU
    python scripts/make_answer.py --method hybrid  # нужен artifacts/dense/e5.index
    python scripts/make_answer.py --method xgb     # BM25+бусты топ-1000 -> XGBRanker
    python scripts/make_answer.py --method bm25 --limit 100  # быстрый прогон

Метод xgb: пайплайн «BM25 с бустами -> переранжированный топ-1000 ->
XGBoost-реранкер -> топ-50». Нужны артефакты обучения
(artifacts/xgb_rerank/model.json + cat_categories.json, см.
experiments/xgb_rerank/), путь можно переопределить --model/--categories.

Формат: query_id,answer — до 50 уникальных item_id через пробел.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "solution"))

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.pipeline.common import (
    apply_metadata_boosts,
    artifacts_dir,
    build_bm25,
    lemmatize_query,
    load_config,
    load_items,
)

MAX_ANSWER = 50


def format_answer(doc_ids: list[str]) -> str:
    """Строка ответа: до MAX_ANSWER уникальных item_id через пробел.

    NB: dict.fromkeys отдаёт dict, по нему нельзя срезать — нужен list().
    """
    unique = list(dict.fromkeys(doc_ids))
    return " ".join(unique[:MAX_ANSWER])


def predict_bm25(cfg: dict, queries: pd.DataFrame) -> dict[str, list[str]]:
    items = load_items(cfg)
    bm25_cfg = cfg["bm25"]
    retriever = build_bm25(items, bm25_cfg)

    predictions = {}
    for row in tqdm(queries.itertuples(), total=len(queries), desc="bm25"):
        tokens = lemmatize_query(row.search_query, row.search_infm_params_text)
        cat_b = bm25_cfg.get("category_boost", 0.0)
        loc_b = bm25_cfg.get("location_boost", 0.0)
        if cat_b > 0 or loc_b > 0:
            predictions[row.query_id] = apply_metadata_boosts(
                retriever,
                tokens,
                items,
                top_k=MAX_ANSWER,
                search_category=row.search_category,
                search_location_id=row.search_location_id,
                category_boost=cat_b,
                location_boost=loc_b,
                pool=int(bm25_cfg.get("boost_pool", 200)),
            )
        else:
            predictions[row.query_id] = [
                h.doc_id for h in retriever.search_tokens(tokens, top_k=MAX_ANSWER)
            ]
    return predictions


def predict_dense(cfg: dict, queries: pd.DataFrame) -> dict[str, list[str]]:
    from src.retrievers.retrieval import DenseRetriever

    index_path = artifacts_dir(cfg) / "dense" / "e5.index"
    if not index_path.exists():
        sys.exit(f"Индекс не найден: {index_path}; сначала scripts/build_dense_index.py")
    dense = DenseRetriever(str(index_path), model_name=cfg["dense"]["model_name"])

    predictions = {}
    for row in tqdm(queries.itertuples(), total=len(queries), desc="dense"):
        ids, _ = dense.search(row.search_query, top_k=MAX_ANSWER)
        predictions[row.query_id] = ids
    return predictions


def predict_hybrid(cfg: dict, queries: pd.DataFrame) -> dict[str, list[str]]:
    from src.retrievers.retrieval import DenseRetriever

    items = load_items(cfg)
    bm25_cfg = cfg["bm25"]
    retriever = build_bm25(items, bm25_cfg)

    index_path = artifacts_dir(cfg) / "dense" / "e5.index"
    if not index_path.exists():
        sys.exit(f"Индекс не найден: {index_path}; сначала scripts/build_dense_index.py")
    dense = DenseRetriever(str(index_path), model_name=cfg["dense"]["model_name"])

    rrf_k = cfg.get("hybrid", {}).get("rrf_k", 60)
    candidates = cfg.get("hybrid", {}).get("candidates_per_retriever", 100)
    predictions = {}
    for row in tqdm(queries.itertuples(), total=len(queries), desc="hybrid"):
        tokens = lemmatize_query(row.search_query, row.search_infm_params_text)
        bm25_hits = retriever.search_tokens(tokens, top_k=candidates)
        dense_hits, _ = dense.search(row.search_query, top_k=candidates)

        scores = {}
        for rank, h in enumerate(bm25_hits, 1):
            scores.setdefault(h.doc_id, 0.0)
            scores[h.doc_id] += 1.0 / (rrf_k + rank)
        for rank, hid in enumerate(dense_hits, 1):
            scores.setdefault(hid, 0.0)
            scores[hid] += 1.0 / (rrf_k + rank)

        top = sorted(scores, key=scores.get, reverse=True)[:MAX_ANSWER]
        predictions[row.query_id] = top
    return predictions


def predict_xgb(
    cfg: dict,
    queries: pd.DataFrame,
    model_path: str,
    categories_path: str,
    depth: int,
) -> dict[str, list[str]]:
    """BM25+бусты -> топ-`depth` -> XGBRanker -> топ-50."""
    from src.rerank.inference import rerank_xgb

    return rerank_xgb(
        cfg, queries,
        model_path=model_path,
        categories_path=categories_path,
        depth=depth,
        show_progress=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Формирование answer.csv для benchmark-запросов."
    )
    parser.add_argument("--method", choices=["bm25", "dense", "xgb", "hybrid"],
                        default="bm25")
    parser.add_argument("--out", default=None,
                        help="путь до output CSV (по умолчанию cfg.paths.answer_dir/answer.csv)")
    parser.add_argument("--limit", type=int, default=None, help="число benchmark-запросов")
    parser.add_argument("--model", default=None,
                        help="model.json XGBRanker (методы xgb)")
    parser.add_argument("--categories", default=None,
                        help="cat_categories.json (методы xgb)")
    parser.add_argument("--depth", type=int, default=None,
                        help="глубина переранжированного BM25-пула (методы xgb; "
                             "по умолчанию cfg.xgb_rerank.depth)")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "config.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir = PROJECT_ROOT / cfg["paths"]["raw_data_dir"]
    queries = pd.read_parquet(raw_dir / "benchmark_queries.parquet")
    if args.limit:
        queries = queries.head(args.limit)
    print(f"Benchmark-запросов: {len(queries)} (limit={args.limit or 'all'})")

    if args.method == "bm25":
        predictions = predict_bm25(cfg, queries)
    elif args.method == "dense":
        predictions = predict_dense(cfg, queries)
    elif args.method == "xgb":
        xcfg = cfg.get("xgb_rerank", {})
        model_path = args.model or str(PROJECT_ROOT / xcfg.get(
            "model", "artifacts/xgb_rerank/model.json"))
        categories_path = args.categories or str(PROJECT_ROOT / xcfg.get(
            "categories", "artifacts/xgb_rerank/cat_categories.json"))
        depth = args.depth or int(xcfg.get("depth", 1000))
        predictions = predict_xgb(cfg, queries, model_path, categories_path, depth)
    else:
        predictions = predict_hybrid(cfg, queries)

    # Гарантии формата: до 50 уникальных item_id на запрос, все запросы покрыты
    answer = pd.DataFrame(
        {
            "query_id": queries["query_id"],
            "answer": [format_answer(predictions.get(q) or []) for q in queries["query_id"]],
        }
    )
    out_path = Path(args.out) if args.out else PROJECT_ROOT / cfg["paths"]["answer_dir"] / "answer.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    answer.to_csv(out_path, index=False)
    print(f"Сохранён ответ: {out_path} ({len(answer)} строк)")


if __name__ == "__main__":
    main()
