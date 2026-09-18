"""CatBoost-переранжировщик поверх BM25-пула: поднимаем релевантные в топ-50.

Идея: Recall@10k чистого BM25 уже 0.9668 (этап A, artifacts/bm25_pool.parquet).
Проблема — поднять релевантные из глубины пула наверх. Пайплайн:

  BM25-пул 10k  ->  CatBoost (фичи запрос×объявление)  ->  топ-1000
               ->  переранжирование bm25-скором с бустом локации  ->  топ-50.

Признаки (query × item): bm25_score/rank/доля ранга, совпадение локации и
категории, лексическое пересечение запроса с заголовком (overlap/coverage),
длины заголовка/описания/запроса, размер пула, is_delivery; категориальные:
microcat_id, search_location_id.

Выборка: сессии валидации делятся детерминированно (порядок validation.parquet
= md5-хеш ключа) — 70% train / 30% eval. Для train-сессий берём все
положительные item'ы пула + негатывы: top-`--neg-top` по bm25 (самые трудные)
+ `--neg-rand` случайных из глубины пула.

Метрики на eval-сессиях (скoring ВСЕГО пула каждой сессии):
  bm25+loc@50     бейзлайн: бустнутый bm25 (как в основном пайплайне);
  cb@50 / cb@1000 CatBoost напрямую;
  cb1000->loc@50  пайплайн из задачи: cb топ-1000 -> bm25*(1+3*loc) -> топ-50;
  cb1000->cb*loc  cb топ-1000, переранжировано cb_score*(1+3*loc).

Запуск из корня проекта:
    python experiments/catboost_rerank/train_reranker.py --pilot   # быстрый пилот
    python experiments/catboost_rerank/train_reranker.py           # полный прогон
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# experiments/catboost_rerank -> корень проекта
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "solution"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

import numpy as np
import pandas as pd
from catboost import Pool
from tqdm import tqdm

from recall_at_k_bm25 import QueryLemmatizer, check_parquet
from src.evaluation.validation import load_validation, validation_relevance
from src.metrics.recall import recall_at_k
from src.pipeline.common import artifacts_dir, load_config, load_items

NUM_FEATURES = [
    "bm25_score", "bm25_rank", "bm25_rank_pct", "loc_match", "cat_match",
    "q_title_overlap", "q_title_coverage", "title_len", "desc_len",
    "query_len", "pool_size", "is_delivery",
]
CAT_FEATURES = ["microcat_id", "search_loc"]
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES


def load_item_info(items: pd.DataFrame) -> dict:
    """item_id -> (loc, cat, microcat, title_frozenset, title_len, desc_len)."""
    info: dict[str, tuple] = {}
    for row in tqdm(items.itertuples(), total=len(items), desc="Индексация корпуса"):
        title = row.title_lem
        if isinstance(title, str):
            title = title.split()
        desc = row.desc_lem
        if isinstance(desc, str):
            desc = desc.split()
        info[row.item_id] = (
            int(row.item_location_id) if row.item_location_id is not None else -1,
            int(row.item_category_id) if row.item_category_id is not None else -1,
            int(row.item_microcat_id) if row.item_microcat_id is not None else -1,
            frozenset(title) if title is not None else frozenset(),
            len(title) if title is not None else 0,
            len(desc) if desc is not None else 0,
        )
    return info


def build_features(
    qrow,
    ids: list[str],
    bm25_scores: np.ndarray,
    qtokens: list[str],
    item_info: dict,
) -> np.ndarray:
    """Матрица признаков (len(ids), len(ALL_FEATURES)); NaN -> -1 для категорий."""
    n = len(ids)
    qset = frozenset(qtokens)
    qlen = max(1, len(qtokens))
    q_loc = qrow.search_location_id
    q_cat = qrow.search_category
    q_loc = int(q_loc) if q_loc is not None and not pd.isna(q_loc) else -1
    q_cat = int(q_cat) if q_cat is not None and not pd.isna(q_cat) else -1
    d_raw = qrow.search_is_delivery_search
    delivery = 0 if (d_raw is None or pd.isna(d_raw)) else int(bool(d_raw))

    X = np.empty((n, len(ALL_FEATURES)), dtype=np.float32)
    ranks = np.arange(n, dtype=np.float32)
    for j, iid in enumerate(ids):
        loc, cat, micro, tset, tlen, dlen = item_info[iid]
        ov = len(qset & tset) if tset else 0
        X[j] = (
            bm25_scores[j], ranks[j], ranks[j] / n,
            float(loc == q_loc), float(cat == q_cat),
            ov, ov / qlen,
            tlen, dlen,
            qlen, n, delivery,
            micro, q_loc if q_loc >= 0 else -1.0,
        )
    return X


def cat_columns(X: np.ndarray) -> np.ndarray:
    """Индексы категориальных колонок в ALL_FEATURES."""
    return np.array([ALL_FEATURES.index(c) for c in CAT_FEATURES], dtype=int)


def to_frame(X: np.ndarray) -> pd.DataFrame:
    """np-матрица признаков -> DataFrame: числовые float32, категориальные int64.

    CatBoost принимает категориальные фичи только типа int/str: чистый
    float-массив трактуется как «без категориальных», и Pool падает с
    CatBoostError «data is numpy array of floating point numerical type...».
    """
    n_num = len(NUM_FEATURES)
    df = pd.DataFrame(X[:, :n_num], columns=NUM_FEATURES, dtype=np.float32)
    for k, name in enumerate(CAT_FEATURES):
        df[name] = X[:, n_num + k].astype(np.int64)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "config.yaml"))
    parser.add_argument("--pilot", action="store_true",
                        help="быстрый прогон: меньше негативов/итераций/eval-сессий")
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--neg-top", type=int, default=200,
                        help="негативы из топа bm25 (самые трудные)")
    parser.add_argument("--neg-rand", type=int, default=300,
                        help="случайные негативы из глубины пула")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--loss", choices=["yetirank", "logloss"], default="yetirank")
    parser.add_argument("--loc-boost", type=float, default=None,
                        help="буст локации на финальном шаге (по умолчанию из конфига)")
    parser.add_argument("--cb-top", type=int, default=1000,
                        help="сколько лучших по CatBoost пропускать в финал")
    parser.add_argument("--eval-chunk", type=int, default=40,
                        help="сколько eval-сессий скорим одним батчем")
    parser.add_argument("--model-out", default=str(Path(__file__).parent / "model.cb"))
    args = parser.parse_args()
    if args.pilot:
        args.neg_top, args.neg_rand = 100, 100
        args.iterations = 300

    cfg = load_config(args.config)
    artifacts = artifacts_dir(cfg)
    check_parquet(artifacts / "validation.parquet", "validation.parquet")
    check_parquet(artifacts / "items_processed.parquet", "items_processed.parquet")
    check_parquet(artifacts / "bm25_pool.parquet", "bm25_pool.parquet")
    loc_boost = (args.loc_boost if args.loc_boost is not None
                 else cfg["bm25"].get("location_boost", 2.0))

    validation = load_validation(artifacts / "validation.parquet")
    relevance = validation_relevance(validation)
    pool = pd.read_parquet(artifacts / "bm25_pool.parquet")
    pool_groups = {qid: (g["item_id"].tolist(), g["bm25_score"].to_numpy())
                   for qid, g in pool.groupby("query_id", sort=False)}
    items = load_items(cfg)
    item_info = load_item_info(items)
    query_tokens = QueryLemmatizer()

    n_train = int(len(validation) * args.train_frac)
    train_val = validation.iloc[:n_train]
    eval_val = validation.iloc[n_train:]
    if args.pilot:
        eval_val = eval_val.head(150)
    print(f"Сессий: train={len(train_val)}, eval={len(eval_val)} | "
          f"loss={args.loss}, neg_top={args.neg_top}, neg_rand={args.neg_rand}, "
          f"iters={args.iterations}, loc_boost={loc_boost}")

    # ------------------------------------------------- train-выборка (сэмпл негативов)
    rng = np.random.default_rng(42)
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    g_parts: list[np.ndarray] = []
    t0 = time.time()
    for gi, qrow in enumerate(tqdm(train_val.itertuples(), total=len(train_val),
                                   desc="Train-фичи")):
        group = pool_groups.get(qrow.query_id)
        if group is None or len(group[0]) == 0:
            continue
        ids, scores = group
        rel = relevance.get(qrow.query_id, set())
        labels = np.fromiter((1.0 if i in rel else 0.0 for i in ids), dtype=np.float32,
                             count=len(ids))
        pos_idx = np.flatnonzero(labels > 0)
        neg_head = np.flatnonzero(labels == 0)[: args.neg_top]
        rest = np.flatnonzero(labels == 0)[args.neg_top:]
        take_rand = min(args.neg_rand, len(rest))
        neg_rand = rng.choice(rest, size=take_rand, replace=False) if take_rand else rest
        sel = np.concatenate([pos_idx, neg_head, neg_rand])
        toks = query_tokens(qrow.search_query, qrow.search_infm_params_text)
        X_parts.append(build_features(qrow, [ids[i] for i in sel], scores[sel],
                                       toks, item_info))
        y_parts.append(labels[sel])
        g_parts.append(np.full(len(sel), gi, dtype=np.int64))

    X_train = np.concatenate(X_parts); y_train = np.concatenate(y_parts)
    g_train = np.concatenate(g_parts)
    del X_parts, y_parts, g_parts
    print(f"Train: {X_train.shape[0]:,} строк, {int(y_train.sum()):,} положительных "
          f"({time.time() - t0:.0f} c)")

    train_pool = Pool(data=to_frame(X_train), label=y_train, group_id=g_train,
                      feature_names=ALL_FEATURES, cat_features=CAT_FEATURES)

    # ------------------------------------------------------------------ обучение
    if args.loss == "yetirank":
        from catboost import CatBoostRanker
        model = CatBoostRanker(iterations=args.iterations, depth=args.depth,
                               learning_rate=args.learning_rate,
                               loss_function="YetiRank", random_seed=42,
                               verbose=100, thread_count=-1,
                               metric_period=100)
    else:
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(iterations=args.iterations, depth=args.depth,
                                   learning_rate=args.learning_rate,
                                   loss_function="Logloss", random_seed=42,
                                   verbose=100, thread_count=-1)
    t0 = time.time()
    model.fit(train_pool)
    print(f"Обучение: {time.time() - t0:.0f} c | итераций={model.tree_count_}")
    model.save_model(args.model_out)
    print(f"Модель сохранена: {args.model_out}")
    try:
        imp = model.get_feature_importance(train_pool)
        order = np.argsort(-imp)
        print("Важность признаков:")
        for i in order:
            print(f"  {ALL_FEATURES[i]:>18s} {imp[i]:6.1f}")
    except Exception as exc:  # noqa: BLE001
        print(f"(важность недоступна: {exc})")

    # ------------------------------------------------------------------- оценка
    variants = ["bm25+loc@50", f"cb@50", f"cb@{args.cb_top}",
                f"cb{args.cb_top}->loc@50", f"cb{args.cb_top}->cb*loc@50"]
    acc = {v: 0.0 for v in variants}
    n_eval = 0
    t0 = time.time()
    eval_rows = list(eval_val.itertuples())
    for ch in tqdm(range(0, len(eval_rows), args.eval_chunk), desc="Eval"):
        chunk = eval_rows[ch: ch + args.eval_chunk]
        feats, metas = [], []
        for qrow in chunk:
            group = pool_groups.get(qrow.query_id)
            if group is None or len(group[0]) == 0:
                continue
            ids, scores = group
            toks = query_tokens(qrow.search_query, qrow.search_infm_params_text)
            feats.append(build_features(qrow, ids, scores, toks, item_info))
            metas.append((qrow, ids, scores))
        if not feats:
            continue
        X = np.concatenate(feats)
        cb = model.predict(to_frame(X)).reshape(-1)
        pos = 0
        for qrow, ids, scores in metas:
            k = len(ids)
            s = cb[pos: pos + k]; pos += k
            rel = relevance.get(qrow.query_id, set())
            if not rel:
                continue
            n_eval += 1
            q_loc = qrow.search_location_id
            q_loc = int(q_loc) if q_loc is not None and not pd.isna(q_loc) else -1
            loc = np.fromiter((item_info[i][0] for i in ids), dtype=np.int64,
                              count=len(ids))
            match = (loc == q_loc)
            boosted = scores * (1.0 + loc_boost * match)

            # бейзлайн: бустнутый bm25 (как в основном пайплайне)
            order = np.argsort(-boosted, kind="stable")
            acc["bm25+loc@50"] += recall_at_k([ids[i] for i in order], rel, k=50)
            # catboost напрямую
            cb_order = np.argsort(-s, kind="stable")
            acc["cb@50"] += recall_at_k([ids[i] for i in cb_order], rel, k=50)
            acc[f"cb@{args.cb_top}"] += recall_at_k([ids[i] for i in cb_order],
                                                     rel, k=args.cb_top)
            # пайплайн: cb топ-1000 -> bm25*(1+3*loc) -> топ-50
            top = cb_order[: args.cb_top]
            fin = np.argsort(-boosted[top], kind="stable")
            acc[f"cb{args.cb_top}->loc@50"] += recall_at_k(
                [ids[int(top[i])] for i in fin], rel, k=50)
            # вариант: cb топ-1000 -> cb_score*(1+3*loc) -> топ-50
            fin2 = np.argsort(-s[top] * (1.0 + loc_boost * match[top]), kind="stable")
            acc[f"cb{args.cb_top}->cb*loc@50"] += recall_at_k(
                [ids[int(top[i])] for i in fin2], rel, k=50)

    print(f"\n== CatBoost rerank поверх BM25-пула, eval={n_eval} сессий "
          f"({time.time() - t0:.0f} c) ==")
    for v in variants:
        print(f"{v:>20s} {acc[v] / n_eval if n_eval else 0.0:>10.4f}")


if __name__ == "__main__":
    main()
