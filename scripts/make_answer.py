"""Формирование ответа answer.csv для benchmark-запросов.

Запуск (production-пайплайн, модель v2):
    python scripts/make_answer.py                 # BM25+бусты -> XGBRanker -> топ-50
    python scripts/make_answer.py --method bm25   # чистый BM25+бусты (бейзлайн)
    python scripts/make_answer.py --limit 100     # быстрый прогон первых 100 запросов

Метод xgb (production): пайплайн «BM25 с бустами -> переранжированный
топ-2000 -> XGBRanker (model_v2) -> топ-50». Модель и категории берутся
из configs/config.yaml (xgb_rerank.model / xgb_rerank.categories),
можно переопределить флагами --model/--categories.

Формат: query_id,answer — до 50 уникальных item_id через пробел.
Ответ детерминирован: фиксированная модель + стабильные сортировки.
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
    parser.add_argument("--method", choices=["bm25", "xgb"],
                        default="xgb",
                        help="xgb = production (BM25+бусты -> XGBRanker v2)")
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
    else:  # xgb — production
        xcfg = cfg.get("xgb_rerank", {})
        model_path = args.model or str(PROJECT_ROOT / xcfg.get(
            "model", "artifacts/xgb_rerank/model_v2.json"))
        categories_path = args.categories or str(PROJECT_ROOT / xcfg.get(
            "categories", "artifacts/xgb_rerank/cat_categories_v2.json"))
        depth = args.depth or int(xcfg.get("depth", 2000))
        predictions = predict_xgb(cfg, queries, model_path, categories_path, depth)

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
