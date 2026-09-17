"""Реализация целевой метрики Recall@k.

    Recall@k(query) = |Top-k(query) ∩ Relevant(query)| / |Relevant(query)|

Итоговая метрика — среднее по всем запросам. Порядок кандидатов внутри
top-k не важен, важен только факт попадания релевантного объявления в набор.
"""

from __future__ import annotations


def recall_at_k(predicted: list[str] | set[str], relevant: set[str], k: int = 50) -> float:
    """Recall@k для одного запроса.

    Args:
        predicted: список item_id, отсортированный по убыванию скора ретривера.
        relevant: множество релевантных item_id (|relevant| > 0).
        k: размер candidate set (в задаче k = 50).

    Returns:
        Доля релевантных объявлений, попавших в первые k предсказаний.
    """
    if not relevant:
        return 0.0
    top_k = set(predicted[:k])
    return len(top_k & relevant) / len(relevant)


def mean_recall_at_k(
    predictions: dict[str, list[str]],
    relevance: dict[str, set[str]],
    k: int = 50,
) -> float:
    """Средний Recall@k по всем запросам.

    Args:
        predictions: {query_id: [item_id, ...]} — предсказания ретривера.
        relevance: {query_id: {item_id, ...}} — эталонные релевантные item_id.
        k: размер candidate set.

    Returns:
        mean(Recall@k) по всем query_id из relevance.
    """
    recalls = [
        recall_at_k(predictions.get(qid, []), rel, k=k)
        for qid, rel in relevance.items()
    ]
    return sum(recalls) / len(recalls) if recalls else 0.0


def per_query_recall_at_k(
    predictions: dict[str, list[str]],
    relevance: dict[str, set[str]],
    k: int = 50,
) -> dict[str, float]:
    """Recall@k по каждому запросу — для анализа и дебага."""
    return {
        qid: recall_at_k(predictions.get(qid, []), rel, k=k)
        for qid, rel in relevance.items()
    }
