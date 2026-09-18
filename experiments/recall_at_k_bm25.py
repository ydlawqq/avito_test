"""Recall@k (k = 50, 250, 500, 1000, 3000, 5000, 10000) для ЧИСТОГО BM25 на всём трейне.

Запуск (CPU; скрипт рассчитан на прогон на другой машине) из корня проекта:
    python experiments/recall_at_k_bm25.py
    python experiments/recall_at_k_bm25.py --size 2000 --out artifacts/recall_at_k_bm25.csv

Что считаем:
  * корпус    — artifacts/items_processed.parquet (все объявления benchmark_items,
                леммы title/params/description уже посчитаны scripts/prepare_data.py);
  * запросы   — ВЕСЬ train.parquet: все уникальные поисковые сессии
                (ключ = search_query + location + delivery + infm_params + category),
                у которых хотя бы один релевантный item есть в корпусе.
                Правила те же, что в src/evaluation/validation.py, но БЕЗ сэмплирования
                до validation.size (2452) — берутся все сессии трейна;
  * ретривер  — ЧИСТЫЙ BM25 (rank_bm25.BM25Okapi) из cfg["bm25"]:
                документ = леммы title_lem × title_weight + params_lem + desc_lem,
                запрос   = леммы search_query + search_infm_params_text.
                Никаких бустов категории/локации (apply_metadata_boosts НЕ вызывается),
                без dense/hybrid — только сырой BM25-скор;
  * метрика   — Recall@k = |Top-k ∩ Relevant| / |Relevant|, среднее по сессиям.
                Для каждого запроса top-10000 достаётся один раз, recall для всех k
                считается кумулятивно (предсказания не копятся в памяти).

Оговорка: сессии, у которых НИ ОДИН релевантный item не входит в корпус
benchmark_items, отбрасываются — иначе recall был бы занижен недостижимыми
объявлениями (та же логика, что в src/evaluation/validation.py).

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

from src.data.preprocessing import get_stopwords
from src.evaluation.validation import build_validation, validation_relevance
from src.metrics.recall import recall_at_k
from src.pipeline.common import (
    artifacts_dir,
    build_bm25,
    lemmatize_query,
    load_config,
    load_items,
)

# k, для которых считается и печатается Recall@k
KS: tuple[int, ...] = (50, 250, 500, 1000, 3000, 5000, 10000)
# Глубина выборки кандидатов за один проход (>= max(KS))
TOP_K_MAX: int = max(KS)
# «Весь трейн»: size для build_validation, заведомо больше числа сессий в трейне
ALL_SESSIONS: int = 1_000_000_000


class QueryLemmatizer:
    """Лемматизация запроса с кэшем.

    Один и тот же search_query / infm_params повторяется в трейне десятки раз,
    а lemmatize_query — чистый Python + pymorphy3, поэтому кэш экономит время.
    """

    def __init__(self) -> None:
        self._stopwords = get_stopwords()
        self._cache: dict[tuple[str, str], list[str]] = {}

    def __call__(self, search_query: str, infm_params: str = "") -> list[str]:
        params = "" if infm_params is None else str(infm_params)
        key = (str(search_query), params)
        tokens = self._cache.get(key)
        if tokens is None:
            tokens = lemmatize_query(key[0], params, self._stopwords)
            self._cache[key] = tokens
        return tokens


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
    """Recall@k чистого BM25 по всем сессиям трейна. Возвращает таблицу {k, recall}."""
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
        f"Запросов (сессий трейна): {len(relevance)} | "
        f"релевантных на запрос: mean={rel_sizes.mean():.3f}, "
        f"max={rel_sizes.max()}, сессий с >1 релевантным: {int((rel_sizes > 1).sum())}"
    )

    recall_sum = {k: 0.0 for k in ks}
    retrieved_sum = 0
    saturated = 0  # сессий, у которых кандидатов набралось >= top_k_max

    if not relevance:
        print("Нет сессий для оценки — проверьте пути к train.parquet / корпусу.")
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

    print("\n== Чистый BM25: Recall@k на всём трейне ==")
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
        help="сколько сессий трейна оценивать (по умолчанию — все сессии)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="если задан, сохранить таблицу Recall@k в CSV по этому пути",
    )
    parser.add_argument(
        "--train",
        default=None,
        help="путь к train.parquet (по умолчанию raw_data/train.parquet из конфига)",
    )
    parser.add_argument(
        "--items",
        default=None,
        help="путь к benchmark_items.parquet (по умолчанию raw_data/benchmark_items.parquet)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir = PROJECT_ROOT / cfg["paths"]["raw_data_dir"]
    train_path = Path(args.train) if args.train else raw_dir / "train.parquet"
    items_path = Path(args.items) if args.items else raw_dir / "benchmark_items.parquet"

    if args.size is None:
        print(f"Валидация: весь train ({train_path}) — все сессии, без сэмплирования")
        size = ALL_SESSIONS
    else:
        print(f"Валидация: train ({train_path}), не более {args.size} сессий")
        size = args.size

    validation = build_validation(
        train_path=train_path,
        items_path=items_path,
        size=size,
        seed=cfg["validation"]["seed"],
    )
    print(f"Сессий трейна с item из корпуса: {len(validation)}")
    print(f"Артефакты: {artifacts_dir(cfg)}")

    result = evaluate_pure_bm25(cfg, validation)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_path, index=False)
        print(f"\nСохранено: {out_path}")


if __name__ == "__main__":
    main()