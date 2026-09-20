"""Recall@k (k = 50, 250, 500, 1000, 3000, 5000, 10000) для ЧИСТОГО BM25
на локальной валидации — том же наборе запросов, что использует scripts/evaluate.py.

Запуск (CPU; скрипт рассчитан на прогон на другой машине) из корня проекта:
    python experiments/recall_at_k_bm25.py
    python experiments/recall_at_k_bm25.py --size 200 --out artifacts/recall_at_k_bm25.csv

Что читаем (raw_data/ НЕ нужен — оба файла лежат в artifacts/):
  * запросы — artifacts/validation.parquet: 2452 сессии из train.parquet
              (query_id, признаки запроса, relevant_items). Это тот же файл,
              что читает scripts/evaluate.py, поэтому цифры сравнимы с его
              прогонами на тех же запросах;
  * корпус  — artifacts/items_processed.parquet (189k объявлений benchmark_items,
              леммы title/params/description уже посчитаны scripts/prepare_data.py).

Полный трейн (26 556 сессий) здесь намеренно НЕ используется: релевантность для
остальных сессий есть только в 490-МБ raw_data/train.parquet, который нужен лишь
там, где строится сама валидация (scripts/prepare_data.py).

Ретривер — ЧИСТЫЙ BM25 (rank_bm25.BM25Okapi) из cfg["bm25"]:
    документ = леммы title_lem × title_weight + params_lem + desc_lem,
    запрос   = леммы search_query + search_infm_params_text.
Никаких бустов категории/локации (apply_metadata_boosts НЕ вызывается),
без dense/hybrid — только сырой BM25-скор.

Метрика — Recall@k = |Top-k ∩ Relevant| / |Relevant|, среднее по сессиям.
Для каждого запроса top-10000 достаётся один раз, recall для всех k считается
кумулятивно (предсказания не копятся в памяти).

Существующий код пайплайна не изменяется — используются готовые функции src/*.
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

from src.evaluation.validation import load_validation, validation_relevance
from src.metrics.recall import recall_at_k
from src.pipeline.common import (
    QueryLemmatizer,  # ре-экспорт: experiments/pool_recall/* импортируют отсюда
    artifacts_dir,
    build_bm25,
    check_parquet,
    load_config,
    load_items,
)

# k, для которых считается и печатается Recall@k
KS: tuple[int, ...] = (50, 250, 500, 1000, 3000, 5000, 10000)
# Глубина выборки кандидатов за один проход (>= max(KS))
TOP_K_MAX: int = max(KS)


# проверка целостности parquet и кэширующая лемматизация запроса — общие
# утилиты, живут в solution/src/pipeline/common.py (используются и обучением)


def top_k_ids(retriever, tokens: list[str], top_k: int) -> list[str]:
    """top-k doc_id по сырому BM25-скору (только score > 0), по убыванию скора.

    Семантика та же, что у BM25Retriever.top_indices (отсекаем нулевые скоры,
    сортируем по убыванию), но скоры считаются один раз: top_indices +
    search_tokens вызывают get_scores дважды на каждый запрос, что на 190k
    документах и десятках тысяч запросов удваивает время прогона.
    """
    scores = retriever.score_tokens(tokens)
    n = len(scores)
    take = min(int(top_k), n)
    if take <= 0:
        return []
    if take < n:
        idx = np.argpartition(-scores, take - 1)[:take]
    else:
        idx = np.arange(n)
    idx = idx[scores[idx] > 0]
    if len(idx) == 0:
        return []
    idx = idx[np.argsort(-scores[idx], kind="stable")]
    return [retriever.doc_ids[int(i)] for i in idx]


def evaluate_pure_bm25(
    cfg: dict,
    validation: pd.DataFrame,
    ks: tuple[int, ...] = KS,
    top_k_max: int = TOP_K_MAX,
) -> pd.DataFrame:
    """Recall@k чистого BM25 по сессиям локальной валидации. Возвращает таблицу {k, recall}."""
    items = load_items(cfg)
    print(f"Корпус: {len(items)} объявлений")

    retriever = build_bm25(items, cfg["bm25"])
    bm25_cfg = cfg["bm25"]
    print(
        "Чистый BM25 (без бустов): "
        f"title_weight={bm25_cfg.get('title_weight')}, "
        f"use_params={bm25_cfg.get('use_params')}, "
        f"use_description={bm25_cfg.get('use_description')}, "
        f"k1={bm25_cfg.get('k1')}, "
        f"epsilon={bm25_cfg.get('epsilon')}"
    )

    relevance = validation_relevance(validation)
    query_tokens = QueryLemmatizer()

    rel_sizes = np.array([len(v) for v in relevance.values()], dtype=np.int64)
    print(
        f"Запросов (сессий валидации): {len(relevance)} | "
        f"релевантных на запрос: mean={rel_sizes.mean():.3f}, "
        f"max={rel_sizes.max()}, сессий с >1 релевантным: {int((rel_sizes > 1).sum())}"
    )

    recall_sum = {k: 0.0 for k in ks}
    retrieved_sum = 0
    saturated = 0  # сессий, у которых кандидатов набралось >= top_k_max

    if not relevance:
        print("Нет сессий для оценки — проверьте пустой validation.parquet.")
        return pd.DataFrame({"k": list(ks), "recall": [0.0] * len(ks)})

    for row in tqdm(
        validation.itertuples(), total=len(validation), desc=f"pure BM25 top-{top_k_max}"
    ):
        rel = relevance[row.query_id]
        tokens = query_tokens(row.search_query, row.search_infm_params_text)
        ranked = top_k_ids(retriever, tokens, top_k_max)
        retrieved_sum += len(ranked)
        if len(ranked) >= top_k_max:
            saturated += 1
        # recall_at_k из src.metrics.recall: |Top-k ∩ Relevant| / |Relevant|
        for k in ks:
            recall_sum[k] += recall_at_k(ranked, rel, k=k)

    n = len(relevance)
    result = pd.DataFrame(
        {"k": list(ks), "recall": [recall_sum[k] / n if n else 0.0 for k in ks]}
    )

    print("\n== Чистый BM25: Recall@k на локальной валидации ==")
    print(f"{'k':>8s} {'Recall@k':>10s}")
    for k, value in result.itertuples(index=False):
        print(f"{k:>8d} {value:>10.4f}")
    print(
        f"\nСессий: {n} | средний размер кандидат-листа: {retrieved_sum / n:.1f} "
        f"| top-{top_k_max} достигнут у {saturated} сессий ({saturated / n:.1%})"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=str(PROJECT_ROOT / "configs" / "config.yaml")
    )
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help="сколько сессий валидации оценивать (по умолчанию — все 2452)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="если задан, сохранить таблицу Recall@k в CSV по этому пути",
    )
    parser.add_argument(
        "--validation",
        default=None,
        help="путь к validation.parquet (по умолчанию artifacts/validation.parquet)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    artifacts = artifacts_dir(cfg)
    val_path = Path(args.validation) if args.validation else artifacts / "validation.parquet"
    items_path = artifacts / "items_processed.parquet"

    # Проверяем целостность входных parquet до тяжёлых операций:
    # validation.parquet даёт запросы и разметку, items_processed.parquet — корпус BM25.
    check_parquet(val_path, "validation.parquet")
    check_parquet(items_path, "items_processed.parquet")

    validation = load_validation(val_path)
    if args.size:
        validation = validation.head(args.size)
    print(f"Сессии валидации: {len(validation)} из {val_path}")
    print(f"Артефакты: {artifacts}")

    result = evaluate_pure_bm25(cfg, validation)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_path, index=False)
        print(f"\nСохранено: {out_path}")


if __name__ == "__main__":
    main()