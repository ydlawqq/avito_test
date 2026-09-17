"""Оценка BM25-ретривера: Recall@k / MRR@k по ground truth в памяти.

Всё выполняется последовательно (без параллелизма). Для каждого запроса из
train-датасета ищем топ-k документов в BM25 и сравниваем с релевантными item_id.

Пример:
    from src.data.evaluate import evaluate_all_variants

    df_metrics = evaluate_all_variants(sample_size=20_000, max_queries=3000)
    print(df_metrics.to_string())
"""

from __future__ import annotations

import pandas as pd

from src.data.bm25_retriever import BM25Retriever
from src.data.prepare import build_bm25_objects
from src.metrics import compute_retrieval_metrics

# Варианты текста запроса, по которым сравниваем.
QUERY_VARIANTS = ["q", "q_params", "q_loc", "q_cat", "q_delivery", "q_all"]


def _prepare_eval_queries(
    queries_df: pd.DataFrame,
    gt: dict[str, list[str]],
    corpus_ids: set[str],
    max_queries: int | None,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Уникальные запросы, у которых есть релевантные документы в корпусе."""
    unique = queries_df.drop_duplicates("query_hash").reset_index(drop=True)
    valid_hashes = {h for h, items in gt.items() if any(i in corpus_ids for i in items)}
    unique = unique[unique["query_hash"].isin(valid_hashes)].reset_index(drop=True)
    if max_queries is not None and len(unique) > max_queries:
        unique = unique.sample(max_queries, random_state=42).reset_index(drop=True)

    gt_slice = {
        h: [i for i in gt[h] if i in corpus_ids]
        for h in unique["query_hash"]
    }
    return unique, gt_slice


def evaluate_variant(
    queries_df: pd.DataFrame,
    corpus: list[dict],
    gt: dict[str, list[str]],
    query_col: str = "q_all",
    k: int = 50,
    max_queries: int | None = 3000,
    ks: tuple[int, ...] = (1, 5, 10, 50),
) -> dict[str, float]:
    """Считает Recall@k и MRR@k для одного варианта запроса (последовательно)."""
    if query_col not in queries_df.columns:
        raise KeyError(f"В queries_df нет колонки '{query_col}'")

    corpus_ids = {d["id"] for d in corpus}
    unique, gt_slice = _prepare_eval_queries(queries_df, gt, corpus_ids, max_queries)
    if unique.empty:
        raise ValueError("Нет запросов с релевантными документами в корпусе.")

    retriever = BM25Retriever(corpus)
    candidates = []
    for q in unique[query_col].fillna("").tolist():
        candidates.append([d.doc_id for d in retriever.search(q, top_k=k)])

    metrics = compute_retrieval_metrics(candidates, gt_slice, ks=ks)
    metrics["n_queries"] = float(len(unique))
    return metrics


def evaluate_all_variants(
    k: int = 50,
    max_queries: int | None = 3000,
    ks: tuple[int, ...] = (1, 5, 10, 50),
    **build_kwargs,
) -> pd.DataFrame:
    """Прогоняет все варианты запроса последовательно на одних и тех же запросах.

    build_kwargs передаются в build_bm25_objects (напр., sample_size, train_path).
    """
    queries_df, corpus, gt = build_bm25_objects(**build_kwargs)

    corpus_ids = {d["id"] for d in corpus}
    unique, gt_slice = _prepare_eval_queries(queries_df, gt, corpus_ids, max_queries)
    if unique.empty:
        raise ValueError("Нет запросов с релевантными документами в корпусе.")

    retriever = BM25Retriever(corpus)
    rows: dict[str, dict[str, float]] = {}
    for col in QUERY_VARIANTS:
        texts = unique[col].fillna("").tolist()
        candidates = []
        for q in texts:
            candidates.append([d.doc_id for d in retriever.search(q, top_k=k)])

        metrics = compute_retrieval_metrics(candidates, gt_slice, ks=ks)
        metrics["n_queries"] = float(len(unique))
        rows[col] = metrics

    df = pd.DataFrame(rows).T
    df.index.name = "query_variant"
    return df