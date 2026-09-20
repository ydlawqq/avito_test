"""Замер Recall@50 замороженной model_v2 на пулах разной глубины (1000/2000/5000).

Вопрос: даёт ли расширение входного пула реранкера (за счёт хвоста бустнутого
BM25) прирост Recall@50 на чистой валидации poolval (2000 сессий из train,
вне старой validation)? Модель НЕ переобучается — только меняется depth.

Пайплайн инференса — ровно тот же код, что в продакшне:
src.rerank.inference.rerank_xgb (BM25 -> бусты -> топ-depth -> XGB -> топ-50).

Заодно считается бейзлайн bm25+loc@50 (голова бустнутого пула) — одинаковый
для всех depth, служит точкой отсчёта.

Запуск из корня проекта:
    python experiments/pool_recall/xgb_depth_eval.py                 # 1000 сессий
    python experiments/pool_recall/xgb_depth_eval.py --limit 100     # smoke
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "solution"))

import pandas as pd

from src.metrics.recall import recall_at_k
from src.pipeline.common import PROJECT_ROOT as ROOT, load_config
from src.rerank.inference import rerank_xgb

DEPTHS = (1000, 2000, 5000)


def recall_of(preds: dict[str, list[str]], relevance: dict[str, set[str]], k: int = 50) -> float:
    vals = []
    for qid, rel in relevance.items():
        if not rel:
            continue
        vals.append(recall_at_k(preds.get(qid) or [], rel, k=k))
    return sum(vals) / max(len(vals), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poolval", default=str(ROOT / "artifacts" / "pool_recall" / "poolval.parquet"))
    parser.add_argument("--model", default=str(ROOT / "artifacts" / "xgb_rerank" / "model_v2.json"))
    parser.add_argument("--categories", default=str(ROOT / "artifacts" / "xgb_rerank" / "cat_categories_v2.json"))
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--depths", default="1000,2000,5000")
    parser.add_argument("--k1", type=float, default=None,
                        help="оверрайд bm25.k1 (иначе из конфига)")
    parser.add_argument("--b", type=float, default=None,
                        help="оверрайд bm25.b (иначе из конфига)")
    args = parser.parse_args()

    cfg = load_config(ROOT / "configs" / "config.yaml")
    if args.k1 is not None:
        cfg["bm25"]["k1"] = args.k1
    if args.b is not None:
        cfg["bm25"]["b"] = args.b
    val = pd.read_parquet(args.poolval)
    if args.limit:
        val = val.head(args.limit)
    relevance = {r.query_id: set(r.relevant_items) for r in val.itertuples()}
    print(f"Сессий: {len(val)}", flush=True)

    depths = tuple(int(d) for d in args.depths.split(","))
    baseline = None
    for depth in depths:
        t0 = time.time()
        preds, base = rerank_xgb(
            cfg, val, args.model, args.categories, depth, collect_baseline=True,
        )
        r_xgb = recall_of(preds, relevance)
        r_base = recall_of(base, relevance)
        baseline = base
        print(f"depth={depth:>5}: xgb@50={r_xgb:.4f}  bm25+loc@50={r_base:.4f}  "
              f"({time.time() - t0:.0f} c)", flush=True)

    assert baseline is not None
    print("\nГотово. xgb@50 сравнивать с bm25+loc@50 (одинаковый для всех depth).")


if __name__ == "__main__":
    main()
