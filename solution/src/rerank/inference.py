"""Инференс XGBoost-реранкера поверх «BM25 с бустами -> топ-`depth`».

Пайплайн:
     чистый BM25 -> сырой пул top-`pool` (cfg["bm25"]["boost_pool"])
                 -> мягкие бусты score * (1 + loc_boost * loc_match)
                 -> топ-`depth` (1000)
                 -> XGBRanker -> финальный топ-50.

Единая точка входа для:
  * scripts/make_answer.py --method xgb    — формирование answer.csv;
  * scripts/evaluate.py    --method xgb    — Recall@k на валидации/holdout.

Один и тот же код => метрика оценки гарантированно соответствует ответу
(расхождение train/inference и дублирующие реализации исключены).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipeline.common import (
    PROJECT_ROOT,
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
    load_pair_stats,
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
# ---------------------------------------------------------------------------

def _load_stage1(cfg: dict, model_path: str, categories_path: str):
    """Загрузить XGBRanker и корпус для stage-1.

    Returns:
        (model, categories, items, item_index, retriever)
    """
    import xgboost as xgb

    model = xgb.XGBRanker()
    model.load_model(model_path)
    # Фичи, на которых модель обучалась (порядок важен). Абляционные модели
    # (например v4_abl без статистических фич) обучены на подмножестве
    # NUMERIC_FEATURES — на инференсе фрейм признаков выравнивается по этому
    # списку в _stage1_scores, иначе XGBoost падает с feature_names mismatch.
    booster = model.get_booster()
    model._expected_features = (
        list(booster.feature_names) if booster.feature_names else None
    )
    categories = load_categories(categories_path)
    print(f"XGBRanker загружен: {model_path} | категорий: "
          f"{ {k: len(v) for k, v in categories.items()} }"
          + (f" | фич: {len(model._expected_features)}"
             if model._expected_features else ""))

    items = prepare_item_side(cfg)
    item_index = ItemSideIndex(items)
    # таблицы парной статистики (гео/локации/microcat/qtext) — обязательны,
    # иначе v2-фичи уйдут в NaN и модель v2 будет деградировать
    stats_dir = cfg.get("xgb_rerank", {}).get("pair_stats")
    if stats_dir:
        load_pair_stats(PROJECT_ROOT / stats_dir)
        print(f"PairStats загружены: {PROJECT_ROOT / stats_dir}")
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
    frame = features_frame(X_num, micro, search_loc, categories)
    expected = getattr(model, "_expected_features", None)
    if expected:
        frame = frame[expected]  # подмножество и порядок — как при обучении
    scores = model.predict(frame)
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

        # порядок по скору модели
        order = np.argsort(-scores, kind="stable")
        top = [
            retriever.doc_ids[int(corpus_idx[int(i)])]
            for i in order[:max_answer]
        ]
        if len(top) < max_answer:
            top = fill_to_max(
                retriever, tokens, items, top, b,
                row.search_category, row.search_location_id, max_answer,
            )
        predictions[row.query_id] = top

        if collect_baseline:
            base = [
                retriever.doc_ids[int(ci)]
                for ci in corpus_idx[:max_answer]
            ]
            if len(base) < max_answer:
                base = fill_to_max(
                    retriever, tokens, items, base, b,
                    row.search_category, row.search_location_id, max_answer,
                )
            baseline[row.query_id] = base

    if collect_baseline:
        return predictions, baseline
    return predictions
