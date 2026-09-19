"""Общие признаки для XGBoost-реранкера поверх BM25-пула с бустами.

Пайплайн (продакшн и обучение совпадают по входу):
    чистый BM25 -> сырой пул top-`pool` (cfg["bm25"]["boost_pool"], 5000)
                -> мягкие бусты score * (1 + loc_boost * loc_match)
                -> топ-`depth` (1000) переранжированных объявлений
                -> XGBoost -> финальный топ-50.

Признаки пары (запрос x объявление). У ЗАПРОСА доступны ТОЛЬКО признаки
`search_*` (как в benchmark_queries.parquet), сторона объявления берётся из
корпуса (benchmark_items.parquet / items_processed.parquet), взаимодействие —
из BM25-скора и лемматизированных текстов.

Один и тот же код используется:
  * experiments/xgb_rerank/build_dataset.py  — формирование обучающего датасета;
  * experiments/xgb_rerank/train_xgb_reranker.py — обучение XGBRanker;
  * scripts/make_answer.py --method xgb      — инференс на benchmark-запросах.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.common import PROJECT_ROOT, load_items

# --------------------------------------------------------------------------
# Спецификация признаков. ПОРЯДОК ФИКСИРОВАН: индексы в матрице = позиции тут.
# --------------------------------------------------------------------------

NUMERIC_FEATURES: list[str] = [
    # --- retrieval-сигналы (вход реранкера) ---
    "bm25_score",        # чистый BM25-скор пары (без бустов)
    "bm25_boosted",      # скор после мягких бустов (то, чем построен топ-1000)
    "bm25_rank_raw",     # ранг в сыром BM25-пуле до бустов (1-based)
    "bm25_rank",         # ранг в переранжированном топе (входной порядок модели)
    "rank_pct",          # bm25_rank / размер пула
    # --- совпадение метаданных запроса (только search_*) с объявлением ---
    "loc_match",         # item_location_id == search_location_id
    "cat_match",         # item_category_id == search_category
    # --- лексическое взаимодействие лемм запроса с полями объявления ---
    "title_overlap",     # |леммы запроса ∩ леммы заголовка|
    "title_coverage",    # title_overlap / len(леммы запроса)
    "title_cov_rev",     # title_overlap / len(леммы заголовка)
    "title_substr",      # сырой запрос — подстрока сырого заголовка (lower)
    "title_startswith",  # заголовок начинается с сырого запроса
    "params_overlap",    # пересечение с леммами item_infm_params_text
    "params_coverage",
    "desc_overlap",      # пересечение с леммами описания
    "desc_coverage",
    # --- длины ---
    "title_len",
    "params_len",
    "desc_len",
    "query_len_lem",
    "query_len_chars",
    "is_delivery",       # search_is_delivery_search
    # --- априорные признаки объявления (из корпуса) ---
    "price_log",         # log1p(item_price); нет цены -> -1
    "has_price",
    "rating",            # item_rating
    "reviews_log",       # log1p(item_rating_reviews_count)
    "phone_hidden",      # item_is_phone_hidden
    "message_forbidden", # item_is_message_forbidden
    "microcat_freq",     # доля корпусов с тем же item_microcat_id
    "loc_freq",          # доля корпусов с тем же item_location_id
]

# Категориальные признаки (XGBoost enable_categorical, dtype pandas "category").
# На инференсе категории фиксируются по обучению (cat_categories.json),
# неизвестные значения становятся NaN -> уходит в missing-ветку дерева.
CAT_FEATURES: list[str] = ["microcat_id", "search_loc"]

FEATURE_NAMES: list[str] = NUMERIC_FEATURES + CAT_FEATURES


def _to_list(value) -> list[str]:
    """Леммы из колонки parquet (list/ndarray) или из строки 'a b c'."""
    if value is None:
        return []
    if isinstance(value, str):
        return value.split() if value else []
    return list(value)


def prepare_item_side(cfg: dict) -> pd.DataFrame:
    """Корпус items_processed.parquet + числовые поля из benchmark_items.parquet.

    items_processed уже содержит леммы (title_lem/params_lem/desc_lem) и
    метаданные; price/rating/флаги лежат только в сыром benchmark_items.
    """
    items = load_items(cfg)
    raw_path = PROJECT_ROOT / cfg["paths"]["raw_data_dir"] / "benchmark_items.parquet"
    extra = pd.read_parquet(
        raw_path,
        columns=[
            "item_id", "item_price", "item_rating", "item_rating_reviews_count",
            "item_is_phone_hidden", "item_is_message_forbidden",
        ],
    )
    return items.merge(extra, on="item_id", how="left")

class ItemSideIndex:
    """Плоские per-item массивы + множества лемм для быстрых фич по парам.

    ВАЖНО: строится из того же DataFrame (в том же порядке), что и BM25-ретривер
    (build_bm25(items, ...)), т.к. индексы корпуса индексируются позиционно.
    """

    def __init__(self, items: pd.DataFrame) -> None:
        self.doc_ids: np.ndarray = items["item_id"].to_numpy()
        n = len(items)

        self.loc = items["item_location_id"].fillna(-1).to_numpy(dtype=np.int64)
        self.cat = items["item_category_id"].fillna(-1).to_numpy(dtype=np.int64)
        self.micro = items["item_microcat_id"].fillna(-1).to_numpy(dtype=np.int64)

        # частоты microcat/локации по корпусу (приоритет популярности)
        micro_counts = pd.Series(self.micro).value_counts()
        loc_counts = pd.Series(self.loc).value_counts()
        self.micro_freq = (
            pd.Series(self.micro).map(micro_counts).to_numpy(dtype=np.float32) / n
        )
        self.loc_freq = (
            pd.Series(self.loc).map(loc_counts).to_numpy(dtype=np.float32) / n
        )

        # цена/рейтинг/флаги (NaN -> безопасные значения)
        price = pd.to_numeric(items["item_price"], errors="coerce")
        self.price_log = np.log1p(price.fillna(0.0).clip(lower=0.0)).to_numpy(
            dtype=np.float32
        )
        self.price_log = np.where(price.isna(), -1.0, self.price_log).astype(np.float32)
        self.has_price = (price > 0).fillna(False).to_numpy(dtype=np.float32)
        self.rating = (
            pd.to_numeric(items["item_rating"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )
        reviews = pd.to_numeric(items["item_rating_reviews_count"], errors="coerce")
        self.reviews_log = np.log1p(reviews.fillna(0.0).clip(lower=0.0)).to_numpy(
            dtype=np.float32
        )
        self.phone_hidden = (
            pd.to_numeric(items["item_is_phone_hidden"], errors="coerce")
            .fillna(0)
            .astype(np.int8)
            .to_numpy()
        )
        self.msg_forbidden = (
            pd.to_numeric(items["item_is_message_forbidden"], errors="coerce")
            .fillna(0)
            .astype(np.int8)
            .to_numpy()
        )

        # множества лемм и сырые заголовки в lower для подстрочных фич
        self.title_sets: list[frozenset[str]] = []
        self.params_sets: list[frozenset[str]] = []
        self.desc_sets: list[frozenset[str]] = []
        self.title_raw_low: list[str] = []
        for row in items.itertuples():
            self.title_sets.append(frozenset(_to_list(row.title_lem)))
            self.params_sets.append(frozenset(_to_list(row.params_lem)))
            self.desc_sets.append(frozenset(_to_list(row.desc_lem)))
            self.title_raw_low.append(str(row.title_raw).lower())

    def check_aligned(self, retriever) -> None:
        """Проверка, что ретривер построен из того же корпуса в том же порядке."""
        if (
            len(retriever.doc_ids) != len(self.doc_ids)
            or retriever.doc_ids[0] != self.doc_ids[0]
            or retriever.doc_ids[-1] != self.doc_ids[-1]
        ):
            raise RuntimeError(
                "BM25-ретривер и ItemSideIndex построены из разных корпусов — "
                "признаки будут неверными. Передавайте один и тот же items-DataFrame."
            )

# --------------------------------------------------------------------------
# Продакшн-пул: BM25 -> сырой топ -> бусты -> топ-depth
# --------------------------------------------------------------------------

def boosted_pool(
    retriever,
    tokens: list[str],
    item_index: ItemSideIndex,
    search_location_id,
    search_category,
    *,
    depth: int = 1000,
    pool: int = 5000,
    loc_boost: float = 2.0,
    cat_boost: float = 0.0,
):
    """Топ-`depth` переранжированного BM25-пула (точная имитация make_answer).

    Логика повторяет apply_metadata_boosts: сырой топ-`pool` по чистому BM25
    (score > 0), умножение скора на (1 + boost * match), сортировка по убыванию
    бустнутого скора, срез depth. Возвращённые corpus_idx выровнены с item_index
    (оба строятся из одного DataFrame — см. check_aligned).

    Returns:
        (corpus_idx, raw_scores, boosted_scores, raw_ranks): позиции в корпусе,
        чистые/бустнутые скоры и ранги в сыром BM25-пуле (1-based). Всё
        отсортировано по убыванию бустнутого скора (входной порядок модели).
    """
    item_index.check_aligned(retriever)
    scores = retriever.score_tokens(tokens)
    take = min(int(pool), len(scores))
    if take <= 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, np.empty(0, np.float32), np.empty(0, np.float32), empty

    idx = np.argpartition(-scores, take - 1)[:take]
    idx = idx[scores[idx] > 0]
    idx = idx[np.argsort(-scores[idx], kind="stable")]
    raw_ranks = np.arange(1, len(idx) + 1, dtype=np.int64)

    mult = np.ones(len(idx), dtype=np.float64)
    if cat_boost > 0 and search_category is not None and not pd.isna(search_category):
        mult *= 1.0 + cat_boost * (item_index.cat[idx] == int(search_category))
    if loc_boost > 0 and search_location_id is not None and not pd.isna(
        search_location_id
    ):
        mult *= 1.0 + loc_boost * (item_index.loc[idx] == int(search_location_id))

    boosted = scores[idx] * mult
    sel = np.argsort(-boosted, kind="stable")[: int(depth)]
    corpus_idx = idx[sel]
    return (
        corpus_idx,
        scores[corpus_idx].astype(np.float32),
        boosted[sel].astype(np.float32),
        raw_ranks[sel],
    )

# --------------------------------------------------------------------------
# Признаки пары
# --------------------------------------------------------------------------

def build_features(
    qrow,
    corpus_idx: np.ndarray,
    raw_scores: np.ndarray,
    boosted_scores: np.ndarray,
    raw_ranks: np.ndarray,
    qtokens: list[str],
    item_index: ItemSideIndex,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Матрица числовых признаков (n, len(NUMERIC_FEATURES)) + категориальные.

    qrow — строка запроса (search_*-признаки), qtokens — леммы запроса
    (search_query + search_infm_params_text, как в BM25-запросе).

    Returns:
        (X_num float32 (n, len(NUMERIC_FEATURES)), microcat int64[n],
         search_loc int64[n])
    """
    n = len(corpus_idx)
    qset = frozenset(qtokens)
    qlen = max(1, len(qtokens))
    q_lower = str(qrow.search_query).lower().strip()

    q_loc = -1
    if qrow.search_location_id is not None and not pd.isna(qrow.search_location_id):
        q_loc = int(qrow.search_location_id)
    q_cat = -1
    if qrow.search_category is not None and not pd.isna(qrow.search_category):
        q_cat = int(qrow.search_category)
    d_raw = qrow.search_is_delivery_search
    delivery = 0 if d_raw is None or pd.isna(d_raw) else int(bool(d_raw))

    pool_n = max(1, n)
    X = np.empty((n, len(NUMERIC_FEATURES)), dtype=np.float32)
    for j, ci in enumerate(corpus_idx):
        tset = item_index.title_sets[ci]
        pset = item_index.params_sets[ci]
        dset = item_index.desc_sets[ci]
        t_ov = len(qset & tset)
        p_ov = len(qset & pset)
        d_ov = len(qset & dset)
        tlen = len(tset)
        title_low = item_index.title_raw_low[ci]
        X[j] = (
            raw_scores[j],                                            # bm25_score
            boosted_scores[j],                                        # bm25_boosted
            raw_ranks[j],                                             # bm25_rank_raw
            j + 1,                                                    # bm25_rank
            (j + 1) / pool_n,                                         # rank_pct
            1.0 if item_index.loc[ci] == q_loc else 0.0,              # loc_match
            1.0 if item_index.cat[ci] == q_cat else 0.0,              # cat_match
            t_ov,                                                     # title_overlap
            t_ov / qlen,                                              # title_coverage
            t_ov / tlen if tlen else 0.0,                             # title_cov_rev
            1.0 if q_lower and q_lower in title_low else 0.0,         # title_substr
            1.0 if q_lower and title_low.startswith(q_lower) else 0.0,
            p_ov,                                                     # params_overlap
            p_ov / qlen,                                              # params_coverage
            d_ov,                                                     # desc_overlap
            d_ov / qlen,                                              # desc_coverage
            tlen,                                                     # title_len
            len(pset),                                                # params_len
            len(dset),                                                # desc_len
            len(qtokens),                                             # query_len_lem
            len(q_lower),                                             # query_len_chars
            delivery,                                                 # is_delivery
            item_index.price_log[ci],                                 # price_log
            item_index.has_price[ci],                                 # has_price
            item_index.rating[ci],                                    # rating
            item_index.reviews_log[ci],                               # reviews_log
            item_index.phone_hidden[ci],                              # phone_hidden
            item_index.msg_forbidden[ci],                             # message_forbidden
            item_index.micro_freq[ci],                                # microcat_freq
            item_index.loc_freq[ci],                                  # loc_freq
        )
    micro = item_index.micro[corpus_idx]
    search_loc = np.full(n, q_loc, dtype=np.int64)
    return X, micro, search_loc

    corpus_idx = idx[sel]
    return (
        corpus_idx,
        scores[corpus_idx].astype(np.float32),
        boosted[sel].astype(np.float32),
        raw_ranks[sel],
    )


def features_frame(
    X_num: np.ndarray,
    micro: np.ndarray,
    search_loc: np.ndarray,
    categories: dict[str, list[int]] | None = None,
) -> pd.DataFrame:
    """DataFrame признаков для XGBRanker: float32-числовые + category-колонки."""
    df = pd.DataFrame(X_num, columns=NUMERIC_FEATURES)
    for col, vals in (("microcat_id", micro), ("search_loc", search_loc)):
        if categories and col in categories:
            df[col] = pd.Categorical(np.asarray(vals), categories=categories[col])
        else:
            df[col] = pd.Categorical(np.asarray(vals))
    return df


# --------------------------------------------------------------------------
# Категории категориальных признаков (фиксируются обучением)
# --------------------------------------------------------------------------

def save_categories(path: str | Path, categories: dict[str, list[int]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({k: [int(x) for x in v] for k, v in categories.items()}, f)


def load_categories(path: str | Path) -> dict[str, list[int]]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {k: [int(x) for x in v] for k, v in raw.items()}

