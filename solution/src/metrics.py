"""
Метрики качества candidate generation: Recall@k и MRR@k.

Соглашения:
- candidates  — список списков item_id для каждого запроса, отсортированный по убыванию
                релевантности (первые элементы — самые релевантные).
- ground_truth — отображение «запрос -> набор релевантных item_id» (например, все позитивные
                пары из тестового датасета, сгруппированные по query_id/query_text).
- Порядок запросов в `candidates` должен совпадать с порядком значений `ground_truth`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

RelType = set[str] | Sequence[str] | Iterable[str]
GTType = Mapping[str, RelType]


def _normalize_gt_queries(candidates: Sequence[Sequence[str]], ground_truth: GTType) -> list[set[str]]:
    """Приводит ground truth к списку множеств, совпадающему по порядку с candidates.

    Запросы без релевантных документов попадают в результат как пустые множества —
    на метрики они влияют через параметр `ignore_empty_gt` в точках вызова.
    """
    n = len(candidates)
    gt_values = list(ground_truth.values())
    if len(gt_values) != n:
        raise ValueError(
            f"Число candidates ({n}) не совпадает с числом запросов в ground_truth "
            f"({len(gt_values)})."
        )
    return [set(v) for v in gt_values]


def recall_at_k(
    candidates: Sequence[Sequence[str]],
    ground_truth: GTType,
    k: int,
    ignore_empty_gt: bool = True,
) -> float:
    """Recall@k: усреднённая по запросам доля релевантных документов в топ-k.

    По определению:
        Recall@k = mean_по_запросам( |топ-k ∩ релевантные| / |релевантные| )

    Например, если у запроса два релевантных объявления, а в топ-k попал только
    одно, то вклад этого запроса равен 1/2.

    Args:
        candidates: ranked списки item_id для каждого запроса.
        ground_truth: {query_id/query_text: релевантные item_id}.
        k: глубина отсечения.
        ignore_empty_gt: если True, запросы без релевантных документов исключаются
            из знаменателя; иначе вклад такого запроса равен 0.

    Returns:
        Recall@k в диапазоне [0.0, 1.0].
    """
    gt = _normalize_gt_queries(candidates, ground_truth)
    score = 0.0
    denominator = 0
    for ranked, rel in zip(candidates, gt):
        if not rel:
            if ignore_empty_gt:
                continue
            denominator += 1
            continue
        denominator += 1
        hits = len(set(ranked[:k]) & rel)
        score += hits / len(rel)
    return score / denominator if denominator else 0.0


def mrr_at_k(
    candidates: Sequence[Sequence[str]],
    ground_truth: GTType,
    k: int,
    ignore_empty_gt: bool = True,
) -> float:
    """MRR@k: средний обратный ранг первой релевантной позиции в топ-k.

    Для каждого запроса: если релевантный документ найден на позиции p (1-based),
    вклад равен 1/p, иначе 0.

    Args:
        candidates: ranked списки item_id для каждого запроса.
        ground_truth: {query_id/query_text: релевантные item_id}.
        k: глубина отсечения.
        ignore_empty_gt: если True, запросы без релевантных документов исключаются
            из знаменателя; иначе для них вклад = 0.

    Returns:
        MRR@k в диапазоне [0.0, 1.0].
    """
    gt = _normalize_gt_queries(candidates, ground_truth)
    rr_sum = 0.0
    denominator = 0
    for ranked, rel in zip(candidates, gt):
        if not rel:
            if ignore_empty_gt:
                continue
            denominator += 1
            continue
        denominator += 1
        for pos, item in enumerate(ranked[:k], start=1):
            if item in rel:
                rr_sum += 1.0 / pos
                break
    return rr_sum / denominator if denominator else 0.0


def compute_retrieval_metrics(
    candidates: Sequence[Sequence[str]],
    ground_truth: GTType,
    ks: Sequence[int] = (10, 50),
    ignore_empty_gt: bool = True,
) -> dict[str, float]:
    """Считает Recall@k и MRR@k сразу для нескольких k.

    Returns:
        Словарь вида {"recall@10": ..., "mrr@10": ..., "recall@50": ..., ...}
    """
    metrics: dict[str, float] = {}
    for k in sorted(set(ks)):
        metrics[f"recall@{k}"] = recall_at_k(candidates, ground_truth, k, ignore_empty_gt)
        metrics[f"mrr@{k}"] = mrr_at_k(candidates, ground_truth, k, ignore_empty_gt)
    return metrics


if __name__ == "__main__":  # быстрая самопроверка без внешних зависимостей
    # a: релевантный документ на 1-м месте           → вклад 1/1
    # b: релевантный документ ровно на 50-й позиции  → вклад 0 при k<50, 1/1 при k=50
    # c: без релевантных документов                  → исключается при ignore_empty_gt=True
    # d: два релевантных, в топ-50 попал только один → вклад 1/2
    cands = [
        ["d1", "d2", "d3"],                                    # a
        [f"x{i}" for i in range(49)] + ["d_gt"],               # b
        ["n1", "n2", "n3"],                                    # c
        ["y1"] + [f"z{i}" for i in range(48)] + ["d2", "d_gt"],  # d
    ]
    gt = {"a": {"d1"}, "b": {"d_gt"}, "c": set(), "d": {"d2", "d_gt"}}

    # k=3: у a найден 1 из 1, у b и d не найден ни один → (1 + 0 + 0) / 3
    assert abs(recall_at_k(cands, gt, k=3) - 1 / 3) < 1e-9, recall_at_k(cands, gt, k=3)
    # k=50: a=1/1, b=1/1, d=1/2 → (1 + 1 + 0.5) / 3 = 5/6
    assert abs(recall_at_k(cands, gt, k=50) - 5 / 6) < 1e-9, recall_at_k(cands, gt, k=50)
    # то же, но c в знаменателе с вкладом 0 → (1 + 1 + 0 + 0.5) / 4
    assert abs(recall_at_k(cands, gt, k=50, ignore_empty_gt=False) - 2.5 / 4) < 1e-9
    # MRR: a=1/1, b=1/50 (позиция 50), d=1/50 (позиция 50)
    assert abs(mrr_at_k(cands, gt, k=3) - 1 / 3) < 1e-9, mrr_at_k(cands, gt, k=3)
    assert abs(mrr_at_k(cands, gt, k=50) - (1 + 1 / 50 + 1 / 50) / 3) < 1e-9

    metrics = compute_retrieval_metrics(cands, gt, ks=(1, 50))
    assert abs(metrics["recall@50"] - 5 / 6) < 1e-9
    assert abs(metrics["mrr@1"] - 1 / 3) < 1e-9
    print("metrics.py: все проверки пройдены OK")
    print(metrics)