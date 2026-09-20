"""Общие компоненты пайплайна: конфиг, загрузка артефактов, сборка BM25-индекса.

Артефакты prepare_data.py (в artifacts/):
  items_processed.parquet — корпус с лемматизированными полями
                            (title_lem, params_lem, desc_lem) + метаданные
  validation.parquet      — локальная валидация с relevant_items
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.preprocessing import get_stopwords, lemmatize_text
from src.retrievers.retrieval import BM25Retriever

# Корень проекта (solution/src/pipeline/ -> ../../../..)
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_config(path: str | Path | None = None) -> dict:
    """Загрузить configs/config.yaml (или переопределённый путь)."""
    cfg_path = Path(path) if path else PROJECT_ROOT / "configs" / "config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def artifacts_dir(cfg: dict) -> Path:
    return PROJECT_ROOT / cfg["paths"]["artifacts_dir"]


def load_items(cfg: dict) -> pd.DataFrame:
    """Загрузить предобработанный корпус объявлений."""
    return pd.read_parquet(artifacts_dir(cfg) / "items_processed.parquet")


def check_parquet(path: Path, what: str) -> None:
    """Проверить, что файл есть и это parquet (magic-байты ``PAR1`` в начале и в конце).

    Большие parquet (items_processed.parquet ~297 МБ) часто обрезаются при
    копировании между машинами: pandas/pyarrow падают с невнятным «Parquet magic
    bytes not found in footer. Either the file is corrupted or this is not a
    parquet file» и вместо пути печатают ``<Buffer>``. Проверка даёт понятную
    ошибку с именем файла и размером до тяжёлой работы.
    """
    if not path.exists():
        raise FileNotFoundError(f"{what} не найден: {path}")
    size = path.stat().st_size
    if size < 8:
        raise ValueError(f"{what} повреждён: {path} — файл пуст или обрезан ({size} байт)")
    with open(path, "rb") as f:
        head = f.read(4)
        f.seek(-4, os.SEEK_END)
        tail = f.read(4)
    if head != b"PAR1" or tail != b"PAR1":
        raise ValueError(
            f"{what} повреждён: {path} ({size} байт, head={head!r}, tail={tail!r}). "
            "Ожидается parquet ('PAR1' ... 'PAR1') — скорее всего файл обрезался при "
            "копировании. Перекопируйте файл целиком или пересоберите артефакты: "
            "python scripts/prepare_data.py."
        )
    print(f"{what}: {path} ({size} байт) — parquet OK")


def build_doc_tokens(
    row,
    title_weight: int = 2,
    use_params: bool = True,
    use_description: bool = True,
) -> list[str]:
    """Собрать документ BM25 из лемм: title (с весом) + params + description.

    Заголовок — самый сильный сигнал соответствия короткому запросу,
    поэтому его леммы дублируются title_weight раз.
    NB: itertuples отдаёт list-колонки как numpy.ndarray — поэтому проверки
    на пустоту делаем через len(), а не через truthiness.
    """

    def _to_list(value) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):  # на случай чтения из parquet как строки
            return value.split() if value else []
        return list(value)

    tokens: list[str] = []
    title = _to_list(row.title_lem)
    for _ in range(max(1, title_weight)):
        tokens.extend(title)
    if use_params:
        tokens.extend(_to_list(row.params_lem))
    if use_description:
        tokens.extend(_to_list(row.desc_lem))
    return tokens


def build_bm25(
    items: pd.DataFrame,
    cfg_bm25: dict,
) -> BM25Retriever:
    """Собрать BM25Retriever из предобработанного корпуса."""
    from tqdm import tqdm

    tokenized = [
        build_doc_tokens(
            row,
            title_weight=cfg_bm25.get("title_weight", 2),
            use_params=cfg_bm25.get("use_params", True),
            use_description=cfg_bm25.get("use_description", True),
        )
        for row in tqdm(items.itertuples(), total=len(items), desc="BM25 corpus")
    ]

    retriever = BM25Retriever(
        doc_ids=items["item_id"].tolist(),
        tokenized_docs=tokenized,
    )
    # Переопределяем параметры BM25Okapi из конфига, если заданы
    k1 = cfg_bm25.get("k1")
    if k1 is not None:
        retriever.bm25.k1 = float(k1)
    b = cfg_bm25.get("b")
    if b is not None:
        retriever.bm25.b = float(b)
    return retriever


def lemmatize_query(
    query: str,
    infm_params: str = "",
    stopwords: frozenset[str] | None = None,
) -> list[str]:
    """Лемматизация запроса (текст + фильтры) — дёшево, выполняется на лету."""
    from src.data.preprocessing import get_stopwords

    sw = stopwords if stopwords is not None else get_stopwords()
    tokens = lemmatize_text(query, sw)
    if infm_params:
        # Параметры фильтра («Вид услуги ...») добавляем без веса
        tokens.extend(lemmatize_text(str(infm_params), sw))
    return tokens


class QueryLemmatizer:
    """Лемматизация запроса с кэшем.

    Один и тот же search_query / infm_params повторяется в трейне десятки раз,
    а lemmatize_query — чистый Python + pymorphy3, поэтому кэш экономит время.
    Используется пакетными скриптами (solution/xgb_rerank/*), на инференсе не нужен.
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


def apply_metadata_boosts(
    retriever: BM25Retriever,
    tokens: list[str],
    items: pd.DataFrame,
    top_k: int,
    search_category: int | None = None,
    search_location_id: int | None = None,
    category_boost: float = 0.0,
    location_boost: float = 0.0,
    pool: int = 200,
    fill_to_top_k: bool = True,
) -> list[str]:
    """BM25 top-k с мягкими бустами за совпадение категории/локации.

    Не жёсткий фильтр (он мог бы необратимо терять релевантные объявления),
    а умножение скора: score * (1 + boost * match).

    pool: размер сырого BM25-пула, внутри которого применяются бусты
    (top-50 достаётся из него). Больше пул — больше шансов, что
    локально-совпадающий документ из глубины ранга поднимется в топ.

    fill_to_top_k: добирать ответ до top_k кандидатами с нулевым скором.
    Нужно для редких/опечатанных запросов (напр. «сабутыльник», «видеоограф»):
    лемма встречается в единицах объявлений, ненулевых скоров < top_k, и без
    добора ответ получается короче 50 (вплоть до пустого). Добор не может
    ухудшить Recall@50: существующий топ только дополняется, не вытесняется.
    """
    scores = retriever.score_tokens(tokens)
    n_take = min(len(scores), max(pool, top_k))
    if n_take == 0:
        return []
    idx = np.argpartition(-scores, n_take - 1)[:n_take]
    idx = idx[scores[idx] > 0]

    cat = (
        items["item_category_id"].to_numpy()[idx]
        if "item_category_id" in items
        else None
    )
    loc = (
        items["item_location_id"].to_numpy()[idx]
        if "item_location_id" in items
        else None
    )

    mult = np.ones(len(idx))
    if category_boost > 0 and search_category is not None and cat is not None:
        mult *= 1.0 + category_boost * (cat == search_category)
    if location_boost > 0 and search_location_id is not None and loc is not None:
        mult *= 1.0 + location_boost * (loc == search_location_id)

    boosted = scores[idx] * mult
    # kind="stable": при равных скорах порядок определяется положением в пуле —
    # результат воспроизводим бит в бит (гарантия детерминизма ответа).
    order = np.argsort(-boosted, kind="stable")[:top_k]
    result = [retriever.doc_ids[int(i)] for i in idx[order]]

    if fill_to_top_k and len(result) < top_k:
        result = _fill_with_zero_score(
            retriever,
            items,
            scores,
            result,
            top_k,
            search_category,
            search_location_id,
            category_boost,
            location_boost,
        )
    return result


def _fill_with_zero_score(
    retriever: BM25Retriever,
    items: pd.DataFrame,
    scores: np.ndarray,
    result: list[str],
    top_k: int,
    search_category: int | None,
    search_location_id: int | None,
    category_boost: float,
    location_boost: float,
) -> list[str]:
    """Дополнить result до top_k, взяв остальные документы по убыванию скора.

    Здесь участвуют и нулевые скоры (документы без общих с запросом лемм):
    порядок среди них задаётся только бустами локации/категории, поэтому
    результат детерминирован (np.argsort стабилен по индексам).
    """
    mult_all = np.ones(len(scores))
    if category_boost > 0 and search_category is not None and "item_category_id" in items:
        cats = items["item_category_id"].to_numpy()
        mult_all *= 1.0 + category_boost * (cats == search_category)
    if location_boost > 0 and search_location_id is not None and "item_location_id" in items:
        locs = items["item_location_id"].to_numpy()
        mult_all *= 1.0 + location_boost * (locs == search_location_id)

    seen = set(result)
    for i in np.argsort(-(scores * mult_all), kind="stable"):
        doc_id = retriever.doc_ids[int(i)]
        if doc_id in seen:
            continue
        seen.add(doc_id)
        result.append(doc_id)
        if len(result) >= top_k:
            break
    return result
