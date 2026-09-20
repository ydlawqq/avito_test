"""Оценка Recall@50 на локальной валидации.

Запуск:
    python scripts/evaluate.py --method bm25   [--limit 500]   # CPU, быстро
    python scripts/evaluate.py --method xgb    [--limit 500]   # XGBoost-реранкер

bm25: перебирает варианты конфигурации (описание on/off, вес заголовка,
бусты категории/локации) и печатает Recall@50 каждого — выбор лучшего
варианта основан только на train-данных.

Валидация построена из train.parquet: только запросы, чьи релевантные
объявления есть в корпусе benchmark_items.parquet (см. src/evaluation/validation.py).

xgb: оценка XGBoost-реранкера (BM25 с бустами -> топ-2000 -> XGBRanker ->
топ-50) ровно тем же кодом, что scripts/make_answer.py (см.
src/rerank/inference.py), поэтому метрика соответствует формируемому ответу.
По умолчанию считается на validation.parquet: если модель обучалась на датасете,
куда эти сессии вошли в train-часть (artifacts/xgb_rerank/dataset*.parquet),
цифра оптимистична (утечка). Честный режим:
    --dataset artifacts/xgb_rerank/dataset_resampled.parquet --split holdout
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "solution"))

import pandas as pd
from tqdm import tqdm

from src.evaluation.validation import load_validation, validation_relevance
from src.metrics.recall import mean_recall_at_k
from src.pipeline.common import (
    apply_metadata_boosts,
    artifacts_dir,
    build_bm25,
    lemmatize_query,
    load_config,
    load_items,
)

# Сетка BM25-вариантов для экспериментов (лучший выбираем по валидации).
# На этой данных: 83% релевантных items лежат в локации поиска -> буст
# локации сильно поднимает recall; категория почти вся одна (114) -> буст
# категории не даёт ничего (проверено экспериментально).
BM25_VARIANTS = [
    # (name, use_description, title_weight, category_boost, location_boost, pool)
    ("tw=3 loc 2.0, pool 1000", True, 3, 0.0, 2.0, 1000),
    ("tw=3 loc 2.0, pool 2000", True, 3, 0.0, 2.0, 2000),
    ("tw=3 loc 2.0, pool 5000", True, 3, 0.0, 2.0, 5000),
    ("tw=4 loc 2.0, pool 2000", True, 4, 0.0, 2.0, 2000),
    ("tw=2 loc 2.0, pool 5000", True, 2, 0.0, 2.0, 5000),
]


def subsample_corpus(
    items: pd.DataFrame, validation: pd.DataFrame, n: int, seed: int = 42
) -> pd.DataFrame:
    """Подвыборка корпуса для быстрых/экономных прогонов.

    Все релевантные items валидации всегда включаются в подвыборку —
    иначе метрика будет занижена; добираем случайными дистракторами до n.
    """
    import numpy as np

    relevant = set()
    for rel in validation["relevant_items"]:
        relevant.update(rel)
    rel_part = items[items["item_id"].isin(relevant)]
    rest = items[~items["item_id"].isin(relevant)]
    n_distract = max(0, n - len(rel_part))
    if n_distract > 0:
        rest = rest.sample(n=n_distract, random_state=seed)
    return pd.concat([rel_part, rest], ignore_index=True)


def evaluate_bm25(
    cfg: dict, validation: pd.DataFrame, limit=None, sample_corpus=None
) -> None:
    """Перебор вариантов BM25 и печать Recall@50."""
    items = load_items(cfg)
    if sample_corpus is not None:
        items = subsample_corpus(items, validation, sample_corpus)
        print(f"Корпус подвыбран до {len(items)} объявлений")

    print("\n== BM25 (перебор вариантов, Recall@50) ==")
    best_name, best_r = None, None
    for name, use_desc, tw, cat_b, loc_b, pool in BM25_VARIANTS:
        cfg["bm25"]["use_description"] = use_desc
        cfg["bm25"]["title_weight"] = tw
        cfg["bm25"]["category_boost"] = cat_b
        cfg["bm25"]["location_boost"] = loc_b
        cfg["bm25"]["boost_pool"] = pool

        retriever = build_bm25(items, cfg["bm25"])
        preds = {row.query_id: [h.doc_id for h in retriever.search_tokens(
            lemmatize_query(row.search_query, row.search_infm_params_text),
            top_k=50,
        )] for row in tqdm(validation.itertuples(), desc=name, total=len(validation))}
        rel = {row.query_id: set(row.relevant_items) for row in validation.itertuples()}
        r = mean_recall_at_k(preds, rel, k=50)
        print(f"  {name:45s}  Recall@50={r:.4f}")
        if best_r is None or r > best_r:
            best_name, best_r = name, r
    print(f"\nЛучший BM25: {best_name}  Recall@50={best_r:.4f}")



def evaluate_xgb(
    cfg: dict,
    validation: pd.DataFrame,
    limit=None,
    *,
    model_path=None,
    categories_path=None,
    depth=None,
    dataset=None,
    split="holdout",
    ks=(50,),
) -> None:
    """XGBoost-реранкер: BM25+бусты -> топ-1000 -> XGBRanker -> топ-50."""
    from src.rerank.inference import rerank_xgb
    from src.rerank.xgb_features import load_categories

    b = cfg["bm25"]
    xcfg = cfg.get("xgb_rerank", {})
    model_path = model_path or str(PROJECT_ROOT / xcfg.get("model", "artifacts/xgb_rerank/model.json"))
    categories_path = categories_path or str(PROJECT_ROOT / xcfg.get("categories", "artifacts/xgb_rerank/cat_categories.json"))
    depth = depth or int(xcfg.get("depth", 1000))

    if dataset:
        from pathlib import Path as _Path
        ds_path = PROJECT_ROOT / dataset
        if not ds_path.exists():
            print(f"Датасет не найден: {ds_path}")
            return
        ds = pd.read_parquet(ds_path)
        if split == "holdout":
            ds = ds[ds["split"] == "holdout"]
        validation = ds
        print(f"Используем датасет как валидацию: {dataset} (split={split}, {len(validation)} запросов)")

    if limit:
        validation = validation.head(limit)

    predictions = rerank_xgb(
        cfg, validation,
        model_path=model_path,
        categories_path=categories_path,
        depth=depth,
        show_progress=True,
    )

    rel = {row.query_id: set(row.relevant_items) for row in validation.itertuples()}
    results = []
    for k in ks:
        r = mean_recall_at_k(predictions, rel, k=k)
        results.append((k, r))
        print(f"  Recall@{k:<4}={' '*4}{r:.4f}")
    return results


def _parse_ks(k, cfg):
    if k is None:
        return (int(cfg.get("recall", {}).get("k", 50)),)
    return tuple(int(x) for x in k.replace(",", " ").split())


def main():
    parser = argparse.ArgumentParser(
        description="Оценка Recall@k на локальной валидации."
    )
    parser.add_argument("--method", choices=["bm25", "xgb"],
                        default="xgb",
                        help="xgb = production (BM25+бусты -> XGBRanker v2)")
    parser.add_argument("--limit", type=int, default=None,
                        help="число валидационных запросов (bm25/xgb)")
    parser.add_argument("--sample-corpus", type=int, default=None,
                        help="оценить BM25 на подвыборке корпуса из N объявлений (анти-OOM режим; "
                             "релевантные items всегда включаются). Финальные цифры — без флага.")
    parser.add_argument("--model", default=None,
                        help="model.json XGBRanker (методы xgb)")
    parser.add_argument("--categories", default=None,
                        help="cat_categories.json (методы xgb)")
    parser.add_argument("--depth", type=int, default=None,
                        help="глубина переранжированного пула (методы xgb)")
    parser.add_argument("--dataset", default=None,
                        help="датасет реранкера для честной оценки без утечки "
                             "(методы xgb; напр. artifacts/xgb_rerank/dataset_resampled.parquet)")
    parser.add_argument("--k", default=None,
                        help="k для Recall@k через запятую/пробел (методы xgb; "
                             "по умолчанию cfg.recall.k=50). Пример: --k 50,100,150")
    parser.add_argument("--split", default="holdout",
                        help="сплит датасета для --dataset: holdout/eval/train "
                             "(методы xgb; holdout — сессии, не участвовавшие в обучении)")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "config.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    validation = load_validation(artifacts_dir(cfg) / "validation.parquet")
    print(f"Валидация: {len(validation)} запросов (limit={args.limit or 'all'})")

    if args.method == "bm25":
        evaluate_bm25(cfg, validation, args.limit, sample_corpus=args.sample_corpus)
    else:  # xgb
        evaluate_xgb(cfg, validation, args.limit, model_path=args.model,
                     categories_path=args.categories, depth=args.depth,
                     dataset=args.dataset, split=args.split,
                     ks=_parse_ks(args.k, cfg))


if __name__ == "__main__":
    main()
