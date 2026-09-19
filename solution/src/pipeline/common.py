"""Общие компоненты пайплайна: конфиг, загрузка артефактов, сборка BM25-индекса.

Артефакты prepare_data.py (в artifacts/):
  items_processed.parquet — корпус с лемматизированными полями
                            (title_lem, params_lem, desc_lem) + метаданные
  validation.parquet      — локальная валидация с relevant_items
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.preprocessing import lemmatize_text
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
    order = np.argsort(-boosted)[:top_k]
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
    for i in np.argsort(-(scores * mult_all)):
        doc_id = retriever.doc_ids[int(i)]
        if doc_id in seen:
            continue
        seen.add(doc_id)
        result.append(doc_id)
        if len(result) >= top_k:
            break
    return result
