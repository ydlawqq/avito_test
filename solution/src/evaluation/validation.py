"""Локальная валидация, построенная только из train.parquet (без benchmark-разметки).

Логика построения:
  1. Из train.parquet оставляем только пары (запрос, item), у которых item_id
     присутствует в корпусе benchmark_items.parquet — иначе нельзя честно
     считать Recall против полного корпуса.
  2. Группируем по полному набору признаков запроса
     (search_query + location + delivery + фильтры + категория) —
     одна группа = один уникальный "поисковый сеанс" с множеством релевантных
     объявлений (все, которые пользователи выбирали по этому запросу).
  3. Детерминированно (seed) сэмплируем не более `size` запросов.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

# Признаки, полностью определяющие поисковый запрос
QUERY_KEY_COLUMNS = [
    "search_query",
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
]

# Колонки запроса, которые остаются в валидации (как в benchmark_queries)
QUERY_OUTPUT_COLUMNS = ["query_id"] + QUERY_KEY_COLUMNS


def build_validation(
    train_path: str | Path,
    items_path: str | Path,
    size: int = 2452,
    seed: int = 42,
) -> pd.DataFrame:
    """Построить валидационную выборку.

    Returns:
        DataFrame с колонками:
            query_id            — детерминированный хеш ключа запроса
            <признаки запроса>
            relevant_items      — list[str] релевантных item_id из корпуса
    """
    train = pd.read_parquet(train_path)
    items = pd.read_parquet(items_path, columns=["item_id"])

    # 1. Оставляем только те пары, чьи объявления есть в корпусе
    corpus_ids = set(items["item_id"])
    train = train[train["item_id"].isin(corpus_ids)]
    train = train[QUERY_KEY_COLUMNS + ["item_id"]].dropna()

    # 2. Группируем по полному ключу запроса
    grouped = (
        train.groupby(QUERY_KEY_COLUMNS, sort=False)["item_id"]
        .apply(lambda s: sorted(set(s)))
        .reset_index(name="relevant_items")
    )

    # 3. Детерминированный сэмпл без numpy RNG: сортировка по md5-хешу ключа
    def _stable_key(row: pd.Series) -> str:
        raw = "\x1f".join(str(row[c]) for c in QUERY_KEY_COLUMNS)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    grouped["_key"] = grouped[QUERY_KEY_COLUMNS].apply(_stable_key, axis=1)
    grouped = grouped.sort_values("_key").head(size).reset_index(drop=True)

    # query_id = первые 16 символов стабильного хеша (уникальны с высокой вероятностью)
    grouped["query_id"] = grouped["_key"].str[:16]

    out = grouped[QUERY_OUTPUT_COLUMNS + ["relevant_items"]]
    return out


def load_validation(path: str | Path) -> pd.DataFrame:
    """Загрузить сохранённую валидацию."""
    return pd.read_parquet(path)


def validation_relevance(validation: pd.DataFrame) -> dict[str, set[str]]:
    """{query_id: set(item_id)} для подсчёта метрики."""
    return {
        row.query_id: set(row.relevant_items) for row in validation.itertuples()
    }
