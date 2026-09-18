"""Оценка Recall@50 на локальной валидации.

Запуск:
    python scripts/evaluate.py --method bm25   [--limit 500]   # CPU, быстро
    python scripts/evaluate.py --method dense  [--limit 500]   # нужен artifacts/dense/e5.index
    python scripts/evaluate.py --method hybrid [--limit 500]

bm25: перебирает варианты конфигурации (описание on/off, вес заголовка,
бусты категории/локации) и печатает Recall@50 каждого — выбор лучшего
варианта основан только на train-данных.

Валидация построена из train.parquet: только запросы, чьи релевантные
объявления есть в корпусе benchmark_items.parquet (см. src/evaluation/validation.py).
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
    if n_distract < len(rest):
        rest = rest.sample(n_distract, random_state=seed)
    out = pd.concat([rel_part, rest]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    print(
        f"Подвыборка корпуса: {len(out)} объявлений "
        f"(из них релевантных: {len(rel_part)})"
    )
    return out


def evaluate_bm25(
    cfg: dict,
    validation: pd.DataFrame,
    limit: int | None,
    sample_corpus: int | None = None,
) -> None:
    """Перебор вариантов BM25.

    sample_corpus: чтобы не поймать OOM на полном корпусе, можно оценить
    варианты на подвыборке корпуса (все релевантные items всегда включаются).
    Финальные цифры снимать на полном корпусе (sample_corpus=None).
    """
    items = load_items(cfg)
    if sample_corpus:
        items = subsample_corpus(items, validation, sample_corpus)

    # ВАЖНО: метрику считаем только по оцениваемым запросам (queries),
    # иначе mean делится на все запросы валидации и занижается при --limit
    queries = validation.head(limit) if limit else validation
    relevance = validation_relevance(queries)
    k = cfg["recall"]["k"]

    qid_to_tokens = {
        row.query_id: lemmatize_query(row.search_query, row.search_infm_params_text)
        for row in queries.itertuples()
    }

    print("\n== BM25: Recall@50 по вариантам ==")
    print(f"{'вариант':38s} {'Recall@50':>10s}")
    import gc

    # Индексы строим ПО ОДНОМУ и освобождаем после варианта — BM25Okapi
    # на полном корпусе держит в памяти ~гигабайты, кэш двух индексов = OOM
    built_keys: set[tuple[bool, int]] = set()
    for vi, (name, use_desc, tw, cat_b, loc_b, pool) in enumerate(BM25_VARIANTS):
        key = (use_desc, tw)
        if key not in built_keys:
            retriever = build_bm25(
                items,
                {"use_description": use_desc, "title_weight": tw, "use_params": True},
            )
            built_keys.add(key)
        # NB: индекс одного key переиспользуется подряд идущими вариантами

        predictions: dict[str, list[str]] = {}
        for row in tqdm(queries.itertuples(), total=len(queries), desc=name, leave=False):
            tokens = qid_to_tokens[row.query_id]
            if cat_b > 0 or loc_b > 0:
                predictions[row.query_id] = apply_metadata_boosts(
                    retriever,
                    tokens,
                    items,
                    top_k=k,
                    search_category=row.search_category,
                    search_location_id=row.search_location_id,
                    category_boost=cat_b,
                    location_boost=loc_b,
                    pool=pool,
                )
            else:
                predictions[row.query_id] = [
                    h.doc_id for h in retriever.search_tokens(tokens, top_k=k)
                ]

        score = mean_recall_at_k(predictions, relevance, k=k)
        print(f"{name:38s} {score:>10.4f}", flush=True)

        # Освобождаем индекс, если следующий вариант использует другой корпус
        next_vi = vi + 1
        if next_vi >= len(BM25_VARIANTS) or (
            BM25_VARIANTS[next_vi][1],
            BM25_VARIANTS[next_vi][2],
        ) != key:
            del retriever
            built_keys.clear()
            gc.collect()


def _load_dense(cfg: dict):
    from src.retrievers.retrieval import DenseRetriever

    index_path = artifacts_dir(cfg) / "dense" / "e5.index"
    if not index_path.exists():
        sys.exit(
            f"Индекс не найден: {index_path}\n"
            "Сначала постройте его: python scripts/build_dense_index.py"
        )
    return DenseRetriever(
        index_path=str(index_path),
        model_name=cfg["dense"]["model_name"],
    )


def evaluate_dense(cfg: dict, validation: pd.DataFrame, limit: int | None) -> None:
    dense = _load_dense(cfg)
    queries = validation.head(limit) if limit else validation
    relevance = validation_relevance(queries)
    k = cfg["recall"]["k"]

    predictions = {}
    for row in tqdm(queries.itertuples(), total=len(queries), desc="dense"):
        # Dense-модель работает на сыром тексте: запрос + фильтры
        query = " ".join(
            x for x in [row.search_query, str(row.search_infm_params_text)] if x
        )
        predictions[row.query_id] = [h.doc_id for h in dense.search(query, top_k=k)]

    print(f"\n== Dense: Recall@50 = {mean_recall_at_k(predictions, relevance, k=k):.4f}")


def evaluate_hybrid(cfg: dict, validation: pd.DataFrame, limit: int | None) -> None:
    from src.retrievers.retrieval import HybridRetriever

    items = load_items(cfg)
    dense = _load_dense(cfg)
    bm25 = build_bm25(items, cfg["bm25"])
    hybrid = HybridRetriever(
        bm25=bm25,
        dense=dense,
        rrf_k=cfg["hybrid"]["rrf_k"],
        dense_weight=cfg["hybrid"]["dense_weight"],
    )

    queries = validation.head(limit) if limit else validation
    relevance = validation_relevance(queries)
    k = cfg["recall"]["k"]
    cpr = cfg["hybrid"]["candidates_per_retriever"]

    predictions = {}
    for row in tqdm(queries.itertuples(), total=len(queries), desc="hybrid"):
        query = " ".join(
            x for x in [row.search_query, str(row.search_infm_params_text)] if x
        )
        tokens = lemmatize_query(row.search_query, row.search_infm_params_text)
        predictions[row.query_id] = [
            h.doc_id
            for h in hybrid.search(
                query, top_k=k, candidates_per_retriever=cpr, query_tokens=tokens
            )
        ]

    print(f"\n== Hybrid: Recall@50 = {mean_recall_at_k(predictions, relevance, k=k):.4f}")

    # Дополнительно: BM25-only на тех же запросах — бенчмарк для сравнения вклада dense
    bm25_only = {}
    for row in queries.itertuples():
        tokens = lemmatize_query(row.search_query, row.search_infm_params_text)
        bm25_only[row.query_id] = [
            h.doc_id for h in bm25.search_tokens(tokens, top_k=k)
        ]
    print(f"   (BM25-only на тех же запросах: {mean_recall_at_k(bm25_only, relevance, k=k):.4f})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["bm25", "dense", "hybrid"], required=True)
    parser.add_argument("--limit", type=int, default=None, help="число запросов валидации")
    parser.add_argument(
        "--sample-corpus",
        type=int,
        default=None,
        help="оценить на подвыборке корпуса из N объявлений (анти-OOM режим; "
        "релевантные items всегда включаются). Финальные цифры — без флага.",
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "config.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    validation = load_validation(artifacts_dir(cfg) / "validation.parquet")
    print(f"Валидация: {len(validation)} запросов (limit={args.limit or 'all'})")

    if args.method == "bm25":
        evaluate_bm25(cfg, validation, args.limit, sample_corpus=args.sample_corpus)
    elif args.method == "dense":
        evaluate_dense(cfg, validation, args.limit)
    else:
        evaluate_hybrid(cfg, validation, args.limit)


if __name__ == "__main__":
    main()
