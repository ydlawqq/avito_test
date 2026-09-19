"""Инференс XGBoost-реранкера поверх «BM25 с бустами -> топ-`depth`».

Два варианта пайплайна (продакшн и оценка совпадают по входу):

1) `rerank_xgb` — один реранкер:
       чистый BM25 -> сырой пул top-`pool` (cfg["bm25"]["boost_pool"])
                   -> мягкие бусты score * (1 + loc_boost * loc_match)
                   -> топ-`depth` (1000)
                   -> XGBRanker -> финальный топ-50;

2) `rerank_xgb_ce` — второй слой реранка кросс-энкодером:
       BM25 + бусты -> топ-`depth` (1000)
                    -> XGBRanker -> топ-`xgb_topk` (150)
                    -> кросс-энкодер (BAAI/bge-reranker-v2-m3) -> топ-50.

Единые точки входа для:
  * scripts/make_answer.py --method xgb    / xgb_ce — формирование answer.csv;
  * scripts/evaluate.py    --method xgb    / xgb_ce — Recall@k на валидации/holdout.

Один и тот же код => метрика оценки гарантированно соответствует ответу
(расхождение train/inference и дублирующие реализации исключены).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipeline.common import (
    apply_metadata_boosts,
    build_bm25,
    lemmatize_query,
)
from src.rerank.xgb_features import (
    ItemSideIndex,
    boosted_pool,
    build_features,
    features_frame,
    load_categories,
    prepare_item_side,
)

MAX_ANSWER = 50


def fill_to_max(
    retriever,
    tokens: list[str],
    items: pd.DataFrame,
    top: list[str],
    bm25_cfg: dict,
    search_category,
    search_location_id,
    max_answer: int = MAX_ANSWER,
) -> list[str]:
    """Добрать ответ до `max_answer` кандидатами BM25 (голова `top` не меняется).

    Нужно для редких/опечатанных запросов («натуропат», «мотопрогулки»): пул
    короче 50, и без добора ответ получается коротким (в predict_bm25 то же
    делает fill_to_top_k). Добор не может ухудшить Recall@50 — существующий
    топ только дополняется, не вытесняется.
    """
    seen = set(top)
    candidates = apply_metadata_boosts(
        retriever,
        tokens,
        items,
        top_k=max_answer,
        search_category=search_category,
        search_location_id=search_location_id,
        category_boost=float(bm25_cfg.get("category_boost", 0.0)),
        location_boost=float(bm25_cfg.get("location_boost", 0.0)),
        pool=int(bm25_cfg.get("boost_pool", 5000)),
    )
    for doc_id in candidates:
        if len(top) >= max_answer:
            break
        if doc_id not in seen:
            seen.add(doc_id)
            top.append(doc_id)
    return top


# ---------------------------------------------------------------------------
# Общий stage-1: BM25+бусты -> топ-`depth` -> XGBRanker
# Используется и одиночным реранкером (rerank_xgb), и связкой с кросс-энкодером
# (rerank_xgb_ce) — вход у обоих вариантов полностью совпадает.
# ---------------------------------------------------------------------------

def _load_stage1(cfg: dict, model_path: str, categories_path: str):
    """Загрузить XGBRanker и корпус для stage-1.

    Returns:
        (model, categories, items, item_index, retriever)
    """
    import xgboost as xgb

    model = xgb.XGBRanker()
    model.load_model(model_path)
    categories = load_categories(categories_path)
    print(f"XGBRanker загружен: {model_path} | категорий: "
          f"{ {k: len(v) for k, v in categories.items()} }")

    items = prepare_item_side(cfg)   # леммы + price/rating/флаги
    item_index = ItemSideIndex(items)
    retriever = build_bm25(items, cfg["bm25"])
    item_index.check_aligned(retriever)  # ретривер и признаки — из одного корпуса
    return model, categories, items, item_index, retriever


def _stage1_scores(
    model,
    categories: dict,
    retriever,
    item_index: ItemSideIndex,
    row,
    tokens: list[str],
    *,
    depth: int,
    pool: int,
    loc_boost: float,
    cat_boost: float,
):
    """BM25+бусты -> топ-`depth` -> XGBRanker -> (corpus_idx, scores).

    Returns:
        (corpus_idx, scores) или (None, None), если у запроса нет документов
        с ненулевым BM25-скором (редкий/опечатанный запрос — вызывающий код
        отдаёт обычный BM25-ответ с добором). scores выровнены с corpus_idx.
    """
    corpus_idx, raw_s, boost_s, raw_r = boosted_pool(
        retriever, tokens, item_index,
        row.search_location_id, row.search_category,
        depth=depth, pool=pool, loc_boost=loc_boost, cat_boost=cat_boost,
    )
    if len(corpus_idx) == 0:
        return None, None
    X_num, micro, search_loc = build_features(
        row, corpus_idx, raw_s, boost_s, raw_r, tokens, item_index,
    )
    scores = model.predict(features_frame(X_num, micro, search_loc, categories))
    return corpus_idx, np.asarray(scores, dtype=np.float64).reshape(-1)


def rerank_xgb(
    cfg: dict,
    queries: pd.DataFrame,
    model_path: str,
    categories_path: str,
    depth: int,
    *,
    collect_baseline: bool = False,
    max_answer: int = MAX_ANSWER,
    show_progress: bool = True,
):
    """XGBRanker поверх переранжированного BM25-топа глубины `depth`.

    Args:
        cfg: конфиг пайплайна (пул/бусты берутся из cfg["bm25"]).
        queries: DataFrame с search_* признаками запросов (и query_id).
        model_path: путь до model.json XGBRanker.
        categories_path: путь до cat_categories.json (те же категории, что при
            обучении — иначе категориальные фичи поедут).
        depth: глубина переранжированного BM25-пула (вход модели).
        collect_baseline: дополнительно вернуть предсказания бейзлайна
            «BM25 с бустами -> топ-50» (голова того же пула, с добором) —
            для сравнения метрик на одних и тех же запросах.
        max_answer: размер итогового ответа (50).

    Returns:
        {query_id: [item_id, ...]} или (predictions, baseline) при
        collect_baseline=True.
    """
    b = cfg["bm25"]
    model, categories, items, item_index, retriever = _load_stage1(
        cfg, model_path, categories_path
    )

    pool = int(b.get("boost_pool", 5000))
    loc_boost = float(b.get("location_boost", 2.0))
    cat_boost = float(b.get("category_boost", 0.0))

    rows = queries.itertuples()
    if show_progress:
        from tqdm import tqdm

        rows = tqdm(rows, total=len(queries), desc="bm25+xgb")

    predictions: dict[str, list[str]] = {}
    baseline: dict[str, list[str]] = {}
    for row in rows:
        tokens = lemmatize_query(row.search_query, row.search_infm_params_text)
        corpus_idx, scores = _stage1_scores(
            model, categories, retriever, item_index, row, tokens,
            depth=depth, pool=pool, loc_boost=loc_boost, cat_boost=cat_boost,
        )
        if corpus_idx is None:
            # редкий/опечатанный запрос: ненулевых BM25-скоров нет -> отдаём
            # обычный BM25-ответ с добором до max_answer (как в predict_bm25)
            predictions[row.query_id] = fill_to_max(
                retriever, tokens, items, [], b,
                row.search_category, row.search_location_id, max_answer,
            )
            if collect_baseline:
                baseline[row.query_id] = list(predictions[row.query_id])
            continue

        order = np.argsort(-scores)[:max_answer]
        # NB: order — позиции в пуле, поэтому doc_ids берём по corpus_idx[order],
        # а не по самим позициям (иначе документы берутся из начала корпуса).
        top = [retriever.doc_ids[int(corpus_idx[int(i)])] for i in order]
        if len(top) < max_answer:
            # пул короче max_answer (редкий запрос) -> добираем BM25-кандидатами
            top = fill_to_max(
                retriever, tokens, items, top, b,
                row.search_category, row.search_location_id, max_answer,
            )
        predictions[row.query_id] = top

        if collect_baseline:
            base = [retriever.doc_ids[int(ci)] for ci in corpus_idx[:max_answer]]
            if len(base) < max_answer:
                base = fill_to_max(
                    retriever, tokens, items, base, b,
                    row.search_category, row.search_location_id, max_answer,
                )
            baseline[row.query_id] = base

    if collect_baseline:
        return predictions, baseline
    return predictions


# ---------------------------------------------------------------------------
# Второй слой реранка: XGBoost -> топ-`xgb_topk` -> кросс-энкодер -> топ-50
# ---------------------------------------------------------------------------

def rerank_xgb_ce(
    cfg: dict,
    queries: pd.DataFrame,
    model_path: str,
    categories_path: str,
    depth: int,
    *,
    xgb_topk: int | None = None,
    ce_model: str | None = None,
    ce_device: str | None = None,
    ce_max_length: int | None = None,
    ce_batch_size: int | None = None,
    ce_local_files_only: bool = False,
    collect_baseline: bool = False,
    max_answer: int = MAX_ANSWER,
    show_progress: bool = True,
    reranker=None,
):
    """BM25+бусты -> топ-`depth` -> XGB -> топ-`xgb_topk` -> кросс-энкодер -> топ-50.

    Args:
        cfg: конфиг пайплайна; пул/бусты — cfg["bm25"], модель кросс-энкодера и
            глубина входа — cfg["cross_encoder"] (аргументы функции их
            переопределяют).
        queries: DataFrame с search_* признаками запросов (и query_id).
        model_path / categories_path / depth: stage-1 (см. rerank_xgb).
        xgb_topk: сколько кандидатов XGBoost отдаёт на вход кросс-энкодеру (150).
        ce_model: имя/путь кросс-энкодера (по умолчанию BAAI/bge-reranker-v2-m3).
        ce_device / ce_max_length / ce_batch_size / ce_local_files_only: параметры
            кросс-энкодера (см. src/rerank/cross_encoder.py).
        collect_baseline: дополнительно вернуть бейзлайны «BM25+бусты -> топ-50» и
            промежуточный «XGB -> топ-50» — сравнение всех слоёв на одних и тех
            же запросах (иначе возвращается один dict).
        max_answer: размер финального ответа (50).
        show_progress: печатать progress bar.
        reranker: готовый CrossEncoderReranker (для переиспользования между
            вызовами; если None — модель загружается здесь).

    Returns:
        {query_id: [item_id, ...]} или
        (predictions, baseline, xgb_only) при collect_baseline=True.
    """
    from src.rerank.cross_encoder import (
        DEFAULT_CE_MODEL,
        CrossEncoderReranker,
        build_item_texts,
        build_query_text,
    )

    b = cfg["bm25"]
    ccfg = cfg.get("cross_encoder", {}) or {}
    # приоритет: явный аргумент -> cfg["cross_encoder"] -> значение по умолчанию
    xgb_topk = int(xgb_topk if xgb_topk is not None else ccfg.get("xgb_topk", 150))
    ce_model = ce_model or ccfg.get("model", DEFAULT_CE_MODEL)
    if ce_device is None:
        ce_device = ccfg.get("device")
    ce_max_length = int(ce_max_length if ce_max_length is not None
                        else ccfg.get("max_length", 512))
    ce_batch_size = int(ce_batch_size if ce_batch_size is not None
                        else ccfg.get("batch_size", 32))
    # описание для кросс-энкодера усекаем так же, как при препроцессинге корпуса
    ce_desc_max_chars = int(
        ccfg.get("desc_max_chars", cfg["preprocessing"]["description_max_chars"])
    )

    if xgb_topk > depth:
        print(f"xgb_topk={xgb_topk} > depth={depth}: вход кросс-энкодера "
              f"ограничен пулом ({depth})")
        xgb_topk = depth
    if xgb_topk < max_answer:
        print(f"ВНИМАНИЕ: xgb_topk={xgb_topk} < max_answer={max_answer}: "
              f"ответ будет добираться BM25-кандидатами")

    # stage-1 (общий с rerank_xgb)
    model, categories, items, item_index, retriever = _load_stage1(
        cfg, model_path, categories_path
    )
    if reranker is None:
        reranker = CrossEncoderReranker(
            ce_model,
            device=ce_device,
            max_length=ce_max_length,
            batch_size=ce_batch_size,
            local_files_only=ce_local_files_only,
        )
    # тексты объявлений в порядке корпуса: corpus_idx индексирует именно этот список
    item_texts = build_item_texts(items, desc_max_chars=ce_desc_max_chars)

    pool = int(b.get("boost_pool", 5000))
    loc_boost = float(b.get("location_boost", 2.0))
    cat_boost = float(b.get("category_boost", 0.0))

    rows = queries.itertuples()
    if show_progress:
        from tqdm import tqdm

        rows = tqdm(rows, total=len(queries), desc=f"bm25+xgb+ce({xgb_topk})")

    predictions: dict[str, list[str]] = {}
    baseline: dict[str, list[str]] = {}
    xgb_only: dict[str, list[str]] = {}
    for row in rows:
        tokens = lemmatize_query(row.search_query, row.search_infm_params_text)
        corpus_idx, scores = _stage1_scores(
            model, categories, retriever, item_index, row, tokens,
            depth=depth, pool=pool, loc_boost=loc_boost, cat_boost=cat_boost,
        )
        if corpus_idx is None:
            # редкий/опечатанный запрос: BM25-ответ с добором (как в predict_bm25)
            answer = fill_to_max(
                retriever, tokens, items, [], b,
                row.search_category, row.search_location_id, max_answer,
            )
            predictions[row.query_id] = answer
            if collect_baseline:
                baseline[row.query_id] = list(answer)
                xgb_only[row.query_id] = list(answer)
            continue

        # --- слой XGB: порядок по скору модели ---
        order_xgb = np.argsort(-scores, kind="stable")
        if collect_baseline:
            xgb_only[row.query_id] = [
                retriever.doc_ids[int(corpus_idx[int(i)])]
                for i in order_xgb[:max_answer]
            ]

        # --- слой кросс-энкодера: только топ-`xgb_topk` кандидатов XGB ---
        cand_pos = order_xgb[:xgb_topk]          # позиции внутри пула
        cand_idx = corpus_idx[cand_pos]          # индексы корпуса
        docs = [item_texts[int(ci)] for ci in cand_idx]
        query_text = build_query_text(row.search_query, row.search_infm_params_text)
        ce_order, _ = reranker.order(query_text, docs)
        top = [retriever.doc_ids[int(cand_idx[int(i)])] for i in ce_order[:max_answer]]
        if len(top) < max_answer:
            # пул короче max_answer (редкий запрос) -> добираем BM25-кандидатами
            top = fill_to_max(
                retriever, tokens, items, top, b,
                row.search_category, row.search_location_id, max_answer,
            )
        predictions[row.query_id] = top

        if collect_baseline:
            base = [retriever.doc_ids[int(ci)] for ci in corpus_idx[:max_answer]]
            if len(base) < max_answer:
                base = fill_to_max(
                    retriever, tokens, items, base, b,
                    row.search_category, row.search_location_id, max_answer,
                )
            baseline[row.query_id] = base

    if collect_baseline:
        return predictions, baseline, xgb_only
    return predictions