"""Пайплайн подготовки BM25-данных полностью в памяти (файлы на диск не пишем).

build_bm25_objects() возвращает три объекта:
    queries_df   — pd.DataFrame: варианты запроса с лемматизированными текстами
                    (q, q_params, q_loc, q_cat, q_delivery, q_all) и признаками search_*;
    corpus       — list[dict]: формат для BM25Retriever: [{"id", "text", "metadata"}, ...];
    ground_truth — dict[str, list[str]]: {query_hash: [отсортированные уникальные item_id]}.

Пример использования:
    from src.data import build_bm25_objects, BM25Retriever

    queries_df, corpus, ground_truth = build_bm25_objects(sample_size=10_000)
    retriever = BM25Retriever(corpus)
    docs = retriever.search(queries_df.loc[0, "q_all"], top_k=50)
"""

from __future__ import annotations

import hashlib
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable

import pandas as pd
from pyarrow.parquet import ParquetFile

from src.data.text_utils import lemmatize_text, normalize_text

# Признаки запроса — только эти колонки доступны для запроса.
QUERY_FEATURES: list[str] = [
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
]

# Текстовые колонки запроса (парные данные train) и текстовые колонки документа (корпус).
QUERY_TEXT_COLUMNS: list[str] = [
    "search_query",
    "search_infm_params_text",
]
ITEM_TEXT_COLUMNS: list[str] = [
    "item_title_raw",
    "item_infm_params_text",
    "item_description_raw",
]

# Доп. колонки, которые попадают в metadata документов корпуса.
METADATA_COLUMNS: list[str] = [
    "item_rating_reviews_count",
    "item_rating",
    "item_price",
    "item_microcat_id",
    "item_longitude",
    "item_location_id",
    "item_latitude",
    "item_is_phone_hidden",
    "item_is_message_forbidden",
    "item_category_id",
]

# Корень репозитория avito_test (parents: data -> src -> solution -> avito_test)
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve_path(path: str | Path) -> Path:
    """Приводит относительный путь к существующему файлу внутри репозитория.

    Сначала ищет относительно корня avito_test, затем относительно cwd.
    """
    p = Path(path)
    if p.is_absolute():
        return p
    for root in (PROJECT_ROOT, Path.cwd()):
        candidate = root / p
        if candidate.exists():
            return candidate
    return p


def _query_key(df: pd.DataFrame) -> pd.Series:
    """Строковый ключ уникального запроса на основе search_query и всех признаков."""
    key = df["search_query"].fillna("").astype(str)
    for col in QUERY_FEATURES:
        key = key + "|" + df[col].fillna("").astype(str)
    return key


def _hash_series(s: pd.Series) -> pd.Series:
    return s.map(lambda k: hashlib.md5(k.encode("utf-8")).hexdigest())


def _lemmatize_chunk(chunk: Iterable[str]) -> dict[str, str]:
    """Worker-функция: лемматизирует набор уникальных нормализованных строк."""
    from pymorphy3 import MorphAnalyzer  # локальный импорт — не грузим в главный процесс

    morph = MorphAnalyzer()
    return {text: lemmatize_text(text, morph) for text in chunk}


def _lemmatize_norm_texts(norm_texts: Iterable[str], workers: int) -> dict[str, str]:
    """Кэш {нормализованная_строка: лемматизированная_строка}, лемматизация в параллели."""
    unique = sorted({t for t in norm_texts if t})
    if not unique:
        return {}
    if workers <= 1:
        return _lemmatize_chunk(unique)
    chunk_size = max(1, len(unique) // workers)
    chunks = [unique[i : i + chunk_size] for i in range(0, len(unique), chunk_size)]
    cache: dict[str, str] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for part in pool.map(_lemmatize_chunk, chunks):
            cache.update(part)
    return cache


def _build_query_variants(df: pd.DataFrame, lem: dict[str, pd.Series]) -> pd.DataFrame:
    """Варианты текста запроса с добавлением признаков (для экспериментов)."""
    q = lem["search_query"]
    params = lem.get("search_infm_params_text", q.mul(0).add(""))

    def _as_int(col: str, fill: int = -1) -> pd.Series:
        return pd.to_numeric(df[col], errors="coerce").fillna(fill).astype(int).astype(str)

    q_loc = q + " loc:" + _as_int("search_location_id")
    q_cat = q + " cat:" + _as_int("search_category")
    q_delivery = q + " delivery:" + _as_int("search_is_delivery_search", fill=-1)

    out = pd.DataFrame(
        {
            "query_hash": _hash_series(_query_key(df)),
            "search_query": df["search_query"],
            "search_infm_params_text": df["search_infm_params_text"],
            "search_location_id": df["search_location_id"],
            "search_is_delivery_search": df["search_is_delivery_search"],
            "search_category": df["search_category"],
            "item_id": df["item_id"],
            "q": q,
            "q_params": (q + " " + params).str.strip(),
            "q_loc": q_loc,
            "q_cat": q_cat,
            "q_delivery": q_delivery,
            "q_all": (q + " " + params + " " + q_loc + " " + q_cat + " " + q_delivery).str.strip(),
        }
    )
    return out


def _build_corpus(
    df: pd.DataFrame,
    lem: dict[str, pd.Series],
    max_item_chars: int | None,
) -> list[dict]:
    """Один документ на уникальный item_id в формате для BM25Retriever."""
    row_positions = df.drop_duplicates("item_id").index
    item_df = df.iloc[row_positions].reset_index(drop=True)

    title = lem["item_title_raw"].iloc[row_positions].reset_index(drop=True)
    params = lem.get("item_infm_params_text", title.mul(0).add("")).iloc[row_positions].reset_index(drop=True)
    desc = lem.get("item_description_raw", title.mul(0).add("")).iloc[row_positions].reset_index(drop=True)

    text = (title + " " + params + " " + desc).str.strip()
    if max_item_chars is not None:
        text = text.str.slice(0, max_item_chars)

    meta_records = (
        item_df[METADATA_COLUMNS].where(pd.notna(item_df[METADATA_COLUMNS]), None).to_dict("records")
    )
    return [
        {"id": str(item_id), "text": text_i, "metadata": meta}
        for item_id, text_i, meta in zip(item_df["item_id"], text, meta_records)
    ]


def _build_ground_truth(df: pd.DataFrame) -> dict[str, list[str]]:
    hashes = _hash_series(_query_key(df))
    grouped = df.assign(query_hash=hashes).groupby("query_hash")["item_id"].agg(
        lambda ids: sorted(set(ids))
    )
    return grouped.to_dict()


def _read_available(path: str | Path, desired: list[str]) -> pd.DataFrame:
    """Читает только колонки из desired, которые реально есть в parquet."""
    available = set(ParquetFile(path).schema_arrow.names)
    columns = list(dict.fromkeys(c for c in desired if c in available))
    return pd.read_parquet(path, columns=columns)


def build_bm25_objects(
    train_path: str | Path = "raw_data/train.parquet",
    corpus_path: str | Path = "raw_data/benchmark_items.parquet",
    workers: int | None = None,
    sample_size: int | None = None,
    seed: int = 42,
    max_item_chars: int | None = 2000,
) -> tuple[pd.DataFrame, list[dict], dict[str, list[str]]]:
    """Строит BM25-объекты в памяти.

    Корпус документов (для поиска) собирается из corpus_path = benchmark_items.parquet
    — это все объявления, среди которых ищем кандидатов. Запросы и ground truth
    строятся из train_path = train.parquet (пары «запрос -> объявление»).

    Args:
        train_path: parquet с парами «запрос -> объявление» (источник запросов и GT).
        corpus_path: parquet с объявлениями-кандидатами (источник документов).
        workers: число процессов для лемматизации (по умолчанию все ядра CPU).
        sample_size: если задан, train-фрейм сэмплируется (быстрые эксперименты).
        seed: random_state для сэмплирования.
        max_item_chars: ограничение длины текста документа в символах.

    Returns:
        (queries_df, corpus, ground_truth) — без записи каких-либо файлов.
    """
    if workers is None:
        workers = os.cpu_count() or 1

    # 1) Запросы и ground truth — из train.
    df = _read_available(
        _resolve_path(train_path), ["item_id", *QUERY_TEXT_COLUMNS, *QUERY_FEATURES]
    )
    if sample_size is not None and len(df) > sample_size:
        df = df.sample(sample_size, random_state=seed)
    df = df.reset_index(drop=True)

    # 2) Корпус документов — из benchmark_items (все объявления-кандидаты).
    corpus_df = _read_available(
        _resolve_path(corpus_path), ["item_id", *ITEM_TEXT_COLUMNS, *METADATA_COLUMNS]
    )
    corpus_df = corpus_df.drop_duplicates("item_id").reset_index(drop=True)

    # 3) Нормализуем и лемматизируем тексты (общий кэш по уникальным строкам).
    norm_q: dict[str, pd.Series] = {
        col: df[col].map(normalize_text) for col in QUERY_TEXT_COLUMNS if col in df.columns
    }
    norm_c_raw: dict[str, pd.Series] = {
        col: corpus_df[col].map(normalize_text)
        for col in ITEM_TEXT_COLUMNS
        if col in corpus_df.columns
    }
    # Обрезаем документы ДО лемматизации: описание бывает на тысячи слов,
    # для BM25 достаточно начала — это экономит почти весь цикл лемматизации.
    if max_item_chars is not None:
        norm_c = {col: s.str.slice(0, max_item_chars) for col, s in norm_c_raw.items()}
    else:
        norm_c = norm_c_raw
    unique_texts: set[str] = set()
    for series in (*norm_q.values(), *norm_c.values()):
        unique_texts.update(series[series != ""].unique())
    cache = _lemmatize_norm_texts(unique_texts, workers)

    lem_q: dict[str, pd.Series] = {
        col: s.map(lambda t: cache.get(t, "")) for col, s in norm_q.items()
    }
    lem_c: dict[str, pd.Series] = {
        col: s.map(lambda t: cache.get(t, "")) for col, s in norm_c.items()
    }

    # 4) Собираем итоговые объекты в памяти.
    queries_df = _build_query_variants(df, lem_q)
    corpus = _build_corpus(corpus_df, lem_c, max_item_chars=max_item_chars)
    ground_truth = _build_ground_truth(df)
    return queries_df, corpus, ground_truth