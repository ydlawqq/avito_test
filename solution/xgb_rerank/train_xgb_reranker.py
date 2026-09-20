"""Обучение XGBoost-реранкера поверх BM25-пула с бустами (топ-1000 -> топ-50).

Вход — датасет из solution/xgb_rerank/build_dataset.py:
    (query_id, item_id, label, split, <30 числовых фич>, microcat_id, search_loc),
где каждая группа query_id — переранжированный топ-1000 продакшн-пайплайна.

Модель: xgboost.XGBRanker, objective rank:ndcg; early stopping по кастомной set-метрике Recall@50 (feval), ndcg@50 — для сравнения.
категориальные признаки (microcat_id, search_loc) через enable_categorical.
Early stopping по ndcg@50 на eval-сплите; финальная метрика — настоящий
Recall@50 по вариантам инференса (сравнимо с experiments/catboost_rerank):
    bm25+loc@50      бейзлайн: порядок бустнутого BM25 (текущий пайплайн);
    xgb@50           порядок по скору XGBRanker;
    xgb<R>->loc@50   топ-R по XGB -> переранжирование бустнутым BM25;
    xgb->rrf<w>@50   RRF-фьюжн рангов XGB и бустнутого BM25.

Дообучение существующей модели: --xgb-model artifacts/xgb_rerank/model.json
(передаётся в fit(xgb_model=...), деревья добавляются к уже натренированным).

Запуск из корня проекта:
    python solution/xgb_rerank/train_xgb_reranker.py                    # полный
    python solution/xgb_rerank/train_xgb_reranker.py --iterations 50    # smoke

Артефакты (artifacts/xgb_rerank/):
    model.json          — модель XGBRanker (load_model);
    cat_categories.json — категории категориальных фич (нужны на инференсе);
    (лог в stdout: метрики, важность признаков).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# solution/xgb_rerank -> корень проекта
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "solution"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm

from src.metrics.recall import recall_at_k
from src.rerank.xgb_features import (
    CAT_FEATURES,
    NUMERIC_FEATURES,
    load_categories,
    save_categories,
)

# варианты финального топ-50 на eval
RERANK_TOPS = (200, 500)     # топ-R по XGB -> переранжирование бустнутым BM25
FUSION_WEIGHTS = (1.0, 2.0, 4.0)  # RRF: 1/(60+r_xgb) + w/(60+r_bm25loc)


def group_sizes(qids: np.ndarray) -> np.ndarray:
    """Размеры групп для подряд идущих одинаковых query_id."""
    if len(qids) == 0:
        return np.empty(0, dtype=np.int64)
    change = np.flatnonzero(qids[1:] != qids[:-1]) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(qids)]])
    return (ends - starts).astype(np.int64)


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Прочитать датасет и проверить, что каждая группа занимает один блок строк."""
    df = pd.read_parquet(path)
    qids = df["query_id"].to_numpy()
    blocks: dict[str, int] = {}
    for qid in qids[np.flatnonzero(np.concatenate(([True], qids[1:] != qids[:-1])))]:
        blocks[qid] = blocks.get(qid, 0) + 1
        if blocks[qid] > 1:
            raise ValueError(
                f"query_id {qid!r} встречается в нескольких блоках строк: группы "
                "должны идти подряд (build_dataset.py пишет их подряд)."
            )
    return df


def build_xy(df: pd.DataFrame, categories: dict[str, list[int]] | None = None,
             numeric_features: list[str] | None = None,
             cat_features: list[str] | None = None):
    num = numeric_features if numeric_features is not None else NUMERIC_FEATURES
    cat = cat_features if cat_features is not None else CAT_FEATURES
    X_num = df[num].to_numpy(dtype=np.float32)
    X = pd.DataFrame(X_num, columns=num)
    for col in cat:
        if categories and col in categories:
            X[col] = pd.Categorical(df[col].to_numpy(), categories=categories[col])
        else:
            X[col] = pd.Categorical(df[col].to_numpy())
    y = df["label"].to_numpy(dtype=np.float32)
    groups = group_sizes(df["query_id"].to_numpy())
    return X, y, groups

def _ranks(order: np.ndarray) -> np.ndarray:
    """order — best-first перестановка; вернуть rank[doc] (1..N)."""
    r = np.empty(len(order), dtype=np.float64)
    r[order] = np.arange(1, len(order) + 1)
    return r


def recall_at50_group(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """1 - Recall@50 одной группы (сессии) — per-group сигнатура sklearn-метрик.

    XGBoost (sklearn-интерфейс, ранкер) оборачивает callable-метрики через
    ltr_metric_decorator и вызывает их отдельно для каждой группы парой
    (labels, predictions), усредняя по группам. Поэтому DMatrix-сигнатура
    тут недопустима — только (y_true, y_score) одной сессии.

    Возвращаем именно ОШИБКУ (1 - Recall@50): early stopping xgboost
    минимизирует метрику, а авто-детект максимизации в 3.x не понимает
    суффиксы — так минимум ошибки гарантированно соответствует максимуму Recall.
    """
    n_rel = float(np.sum(y_true))
    if n_rel <= 0:
        # сессия без релевантных в пуле — Recall@50 = 0 (как в честной метрике)
        return 1.0
    k = min(50, len(y_true))
    top = np.argpartition(-y_score, k - 1)[:k]
    return 1.0 - float(y_true[top].sum()) / n_rel


# имя функции = имя метрики в evals_result. ВАЖНО: xgboost 3.x НЕ понимает
# суффикс "-max" (авто-детект максимизации знает только префиксы auc/map/ndcg),
# поэтому возвращаем ОШИБКУ (1 - Recall@50): её минимизация = максимизация Recall
recall_at50_group.__name__ = "recall@50-err"


def eval_variants(
    booster_scores: np.ndarray,
    boosted: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> dict[str, float]:
    """Recall@50 вариантов инференса по eval-группам.

    booster_scores — скор XGB по каждой строке (в порядке строк датасета),
    boosted — bm25_boosted (порядок продакшн-пула), labels — метки.
    """
    acc: dict[str, float] = {}
    counts: dict[str, int] = {
        "bm25+loc@50": 0,
        "xgb@50": 0,
        **{f"xgb{r}->loc@50": 0 for r in RERANK_TOPS},
        **{f"xgb->rrf{w:g}@50": 0 for w in FUSION_WEIGHTS},
    }
    pos = 0
    for g in groups:
        s_x = booster_scores[pos: pos + g]
        s_b = boosted[pos: pos + g]
        y = labels[pos: pos + g]
        pos += g
        n_rel = y.sum()
        if n_rel <= 0:
            continue
        ids = np.arange(g)
        rel_ids = set(int(i) for i in ids[y > 0])
        order_b = np.argsort(-s_b, kind="stable")
        order_x = np.argsort(-s_x, kind="stable")

        acc["bm25+loc@50"] = acc.get("bm25+loc@50", 0.0) + recall_at_k(
            [int(i) for i in order_b], rel_ids, k=50
        )
        counts["bm25+loc@50"] += 1
        acc["xgb@50"] = acc.get("xgb@50", 0.0) + recall_at_k(
            [int(i) for i in order_x], rel_ids, k=50
        )
        counts["xgb@50"] += 1

        for r in RERANK_TOPS:
            top = order_x[: min(r, g)]
            fin = top[np.argsort(-s_b[top], kind="stable")]
            name = f"xgb{r}->loc@50"
            acc[name] = acc.get(name, 0.0) + recall_at_k(
                [int(i) for i in fin], rel_ids, k=50
            )
            counts[name] += 1

        # rank каждого документа по обоим спискам (документ -> место, 1..N)
        r_x = _ranks(order_x)
        r_b = _ranks(order_b)
        for w in FUSION_WEIGHTS:
            fused = 1.0 / (60.0 + r_x) + w / (60.0 + r_b)
            order_f = np.argsort(-fused, kind="stable")
            name = f"xgb->rrf{w:g}@50"
            acc[name] = acc.get(name, 0.0) + recall_at_k(
                [int(i) for i in order_f], rel_ids, k=50
            )
            counts[name] += 1

    return {name: acc[name] / counts[name] for name in counts if counts[name]}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        default=str(PROJECT_ROOT / "artifacts" / "xgb_rerank" / "dataset.parquet"),
    )
    parser.add_argument("--iterations", type=int, default=2000,
                        help="максимальное число деревьев (реальное — с early stopping)")
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--eta", type=float, default=0.05)
    parser.add_argument("--early-stopping", type=int, default=100)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample", type=float, default=0.8)
    parser.add_argument("--objective", default="rank:ndcg",
                        choices=["rank:ndcg", "rank:pairwise"])
    parser.add_argument("--xgb-model", default=None,
                        help="готовая model.json — ДОобучить поверх неё (fit xgb_model)")
    parser.add_argument("--categories", default=None,
                        help="cat_categories.json для дообучения (иначе выведется из датасета)")
    parser.add_argument("--model-out",
                        default=str(PROJECT_ROOT / "artifacts" / "xgb_rerank" / "model.json"))
    parser.add_argument("--categories-out",
                        default=str(PROJECT_ROOT / "artifacts" / "xgb_rerank" / "cat_categories.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=None,
                        help="потоки XGBoost; по умолчанию — все ядра. "
                             "НЕ ставьте -1: декоратор ltr-метрик xgboost "
                             "передаёт это значение в ThreadPoolExecutor "
                             "и падает с max_workers must be greater than 0")
    parser.add_argument("--exclude-features", default="",
                        help="запятая-разделённый список фич для абляции "
                             "(исключаются из train/eval/holdout), например: "
                             "item_pos_cnt_log,loc_pair_prob")
    args = parser.parse_args()

    excl = [s.strip() for s in args.exclude_features.split(",") if s.strip()]
    num_feats = [c for c in NUMERIC_FEATURES if c not in excl]
    cat_feats = [c for c in CAT_FEATURES if c not in excl]
    if excl:
        dropped = [c for c in NUMERIC_FEATURES + CAT_FEATURES if c in excl]
        print(f"АБЛЯЦИЯ: исключены фичи ({len(dropped)}): {dropped}")

    t0 = time.time()
    df = load_dataset(args.dataset)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    eval_df = df[df["split"] == "eval"].reset_index(drop=True)
    holdout_df = None
    if "holdout" in set(df["split"].unique()):
        holdout_df = df[df["split"] == "holdout"].reset_index(drop=True)
    # train-группы без позитивов (позитив не доехал до топа-1000): внутри них
    # все лейблы 0 -> лямбда-градиенты нулевые, ранжирующий лосс ничему не учит
    pos_per_q = train_df.groupby("query_id")["label"].transform("sum")
    n_drop = int((pos_per_q == 0).sum())
    if n_drop:
        train_df = train_df[pos_per_q > 0].reset_index(drop=True)
        print(f"Train-групп без позитивов отброшено: {n_drop} строк "
              f"(в ранжирующем лоссе они бесполезны)")
    print(f"Датасет: {args.dataset}")
    print(f"  train: {len(train_df):,} строк / {train_df['query_id'].nunique()} сессий, "
          f"позитивов {int(train_df['label'].sum()):,}")
    print(f"  eval:  {len(eval_df):,} строк / {eval_df['query_id'].nunique()} сессий, "
          f"позитивов {int(eval_df['label'].sum()):,}")
    if holdout_df is not None:
        print(f"  holdout: {len(holdout_df):,} строк / "
              f"{holdout_df['query_id'].nunique()} сессий, "
              f"позитивов {int(holdout_df['label'].sum()):,}")

    # категории фиксируем по ВСЕМУ датасету (train+eval); на инференсе
    # категории из этого файла, unseen-значения уходят в missing-ветку дерева
    categories = load_categories(args.categories) if args.categories else None
    if categories is None:
        categories = {
            col: sorted(df[col].dropna().unique().tolist()) for col in CAT_FEATURES
        }
        save_categories(args.categories_out, categories)
        print(f"Категории сохранены: {args.categories_out} "
              f"({ {k: len(v) for k, v in categories.items()} })")

    X_tr, y_tr, g_tr = build_xy(train_df, categories, num_feats, cat_feats)
    X_ev, y_ev, g_ev = build_xy(eval_df, categories, num_feats, cat_feats)

    model = xgb.XGBRanker(
        objective=args.objective,
        # ndcg@50 — прокси для сравнения, recall@50-max — целевая метрика:
        # early stopping работает по ПОСЛЕДНЕЙ метрике списка
        eval_metric=["ndcg@50", recall_at50_group],
        tree_method="hist",
        enable_categorical=True,
        max_depth=args.max_depth,
        learning_rate=args.eta,
        n_estimators=args.iterations,
        early_stopping_rounds=args.early_stopping,
        subsample=args.subsample,
        colsample_bytree=args.colsample,
        random_state=args.seed,
        n_jobs=args.n_jobs,
    )
    print(f"Обучение: objective={args.objective}, depth={args.max_depth}, "
          f"eta={args.eta}, trees<={args.iterations}, "
          f"early_stopping={args.early_stopping}"
          + (f", xgb_model={args.xgb_model}" if args.xgb_model else ""))
    model.fit(
        X_tr, y_tr,
        group=g_tr,
        eval_set=[(X_ev, y_ev)],
        eval_group=[g_ev],
        verbose=100,
        xgb_model=args.xgb_model,
    )
    best = getattr(model, "best_iteration", None)
    n_trees = model.n_estimators if best is None else best + 1
    print(f"Деревьев: {n_trees} (best_iteration={best}), {time.time() - t0:.0f} c")
    evals = model.evals_result()
    ndcg = evals["validation_0"]["ndcg@50"]
    rec_err = evals["validation_0"]["recall@50-err"]
    if best is not None and best < len(ndcg):
        print(f"ndcg@50 на eval: best={ndcg[best]:.5f} (последний={ndcg[-1]:.5f})")
        # recall@50-err = 1 - Recall@50: минимум ошибки = максимум Recall
        print(f"recall@50 на eval (early stopping): best={1 - rec_err[best]:.5f} "
              f"(последний={1 - rec_err[-1]:.5f})")

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    model.save_model(args.model_out)
    print(f"Модель сохранена: {args.model_out}")

    # --------------------------------------------------------------- eval
    scores = model.predict(X_ev).astype(np.float64)
    variants = eval_variants(
        scores,
        eval_df["bm25_boosted"].to_numpy(),
        eval_df["label"].to_numpy(),
        g_ev,
    )
    print(f"\n== Recall@50 на eval-сессиях ({eval_df['query_id'].nunique()}) ==")
    for name, value in variants.items():
        print(f"{name:>18s} {value:10.4f}")

    if holdout_df is not None:
        # holdout — полные пулы сессий, не участвовавших ни в train,
        # ни в early stopping: самая честная финальная метрика
        X_h, _, g_h = build_xy(holdout_df, categories, num_feats, cat_feats)
        h_variants = eval_variants(
            model.predict(X_h).astype(np.float64),
            holdout_df["bm25_boosted"].to_numpy(),
            holdout_df["label"].to_numpy(),
            g_h,
        )
        print(f"\n== Recall@50 на holdout-сессиях, полные пулы "
              f"({holdout_df['query_id'].nunique()}) ==")
        for name, value in h_variants.items():
            print(f"{name:>18s} {value:10.4f}")

    try:
        imp = model.get_booster().get_score(importance_type="gain")
        order = sorted(imp.items(), key=lambda kv: -kv[1])[:20]
        print("\nВажность признаков (gain, топ-20):")
        for name, gain in order:
            print(f"  {name:>18s} {gain:10.1f}")
    except Exception as exc:  # noqa: BLE001
        print(f"(важность недоступна: {exc})")


if __name__ == "__main__":
    main()

