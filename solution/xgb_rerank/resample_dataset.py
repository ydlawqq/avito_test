"""Пересэмплирование готового dataset.parquet без пересборки BM25-пулов.

Зачем: датасет был собран с --neg-frac 0, из-за чего train-сессии содержат
только ~50 головных строк пула — модель не видит ранги 51-1000 и не может
научиться вытаскивать глубокие позитивы. Полная пересборка с хвостами стоит
~2.5 ч, а ПОЛНЫЕ пулы уже лежат в eval-части файла — их можно переиспользовать
как новые train-сессии.

Новый сплит (eval-сессии делятся по порядку следования в файле):
    train   = старый train (головы, ~50 строк/сессия) + первые EVAL_TRAIN_FRAC
              eval-сессий с ПОЛНЫМИ пулами (источник глубоких негативов/позитивов);
    eval    = следующая доля eval-сессий, строки сэмплируются
              (все позитивы + первые EVAL_HEAD строк + EVAL_TAIL_RAND случайных
              из хвоста) — быстрый early stopping;
    holdout = оставшиеся eval-сессии с ПОЛНЫМИ пулами — честная финальная
              метрика Recall@50 на полных пулах.

Хвосты СТАРЫХ train-сессий невосстановимы (не сохранялись при сборке) —
их головы используются как дополнительный train-сигнал упорядочивания головы.
Утечки нет: holdout-сессии не участвуют ни в train, ни в early stopping.

Запуск из корня проекта:
    python solution/xgb_rerank/resample_dataset.py
    python solution/xgb_rerank/resample_dataset.py \
        --limit-eval-sessions 50     # smoke: только первые eval-сессии
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "solution"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

EVAL_TRAIN_FRAC = 0.60   # доля eval-сессий, переходящих в train с полными пулами
EVAL_FAST_FRAC = 0.50    # от остатка: доля на быстрый eval (остальное — holdout)
EVAL_HEAD = 100          # голова быстрого eval, сохраняется целиком
EVAL_TAIL_RAND = 150     # случайных строк хвоста быстрого eval
SEED = 42


def session_bounds(qids: np.ndarray):
    """Границы непрерывных блоков одинаковых query_id (группы идут подряд)."""
    change = np.flatnonzero(qids[1:] != qids[:-1]) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(qids)]])
    return starts, ends



def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        default=str(PROJECT_ROOT / "artifacts" / "xgb_rerank" / "dataset.parquet"),
    )
    parser.add_argument(
        "--out",
        default=str(
            PROJECT_ROOT / "artifacts" / "xgb_rerank" / "dataset_resampled.parquet"
        ),
    )
    parser.add_argument(
        "--limit-eval-sessions", type=int, default=None,
        help="smoke: обработать только первые N eval-сессий",
    )
    args = parser.parse_args()

    t0 = time.time()
    print(f"Чтение {args.dataset} ...", flush=True)
    df = pd.read_parquet(args.dataset)
    print(f"  {len(df):,} строк, {time.time() - t0:.0f} c", flush=True)

    qids = df["query_id"].to_numpy()
    starts, ends = session_bounds(qids)
    split_col = df["split"].to_numpy()

    # старый train переносится как есть; границы eval-сессий
    train_idx_old = np.flatnonzero(split_col == "train")
    eval_bounds = [(s, e) for s, e in zip(starts, ends) if split_col[s] == "eval"]
    if args.limit_eval_sessions:
        eval_bounds = eval_bounds[: args.limit_eval_sessions]
    n_eval = len(eval_bounds)
    print(f"Старый train: {len(train_idx_old):,} строк | "
          f"eval-сессий к обработке: {n_eval}", flush=True)

    rng = np.random.default_rng(SEED)
    n_tr = int(n_eval * EVAL_TRAIN_FRAC)
    n_rest = n_eval - n_tr
    n_fast = int(n_rest * EVAL_FAST_FRAC)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # схема файла-источника; кастование при записи обязательно, т.к. pandas 3
    # отдаёт large_string, а в parquet-файле string (иначе ParquetWriter падает
    # с "Table schema does not match schema used to create file")
    target_schema = pq.ParquetFile(args.dataset).schema_arrow
    writer = pq.ParquetWriter(out_path, target_schema, compression="zstd")

    stats: dict[str, dict] = {}

    def emit(part: pd.DataFrame, split: str) -> None:
        if part.empty:
            return
        part = part.copy()
        part["split"] = split
        table = pa.Table.from_pandas(part, preserve_index=False)
        if table.schema != target_schema:
            table = table.cast(target_schema)
        writer.write_table(table)
        st = stats.setdefault(split, {"rows": 0, "pos": 0, "sessions": 0})
        st["rows"] += len(part)
        st["pos"] += int(part["label"].sum())
        st["sessions"] += part["query_id"].nunique()

    # 1) старый train — как есть (сигнал упорядочивания головы)
    emit(df.iloc[train_idx_old], "train")
    print(f"старый train перенесён: {len(train_idx_old):,} строк", flush=True)

    # 2) eval-сессии: n_tr -> train (полные пулы), далее fast/holdout
    for i, (s, e) in enumerate(eval_bounds):
        if i < n_tr:
            emit(df.iloc[s:e], "train")
            continue
        block = df.iloc[s:e]
        if i < n_tr + n_fast:
            # быстрый eval: все позитивы + голова + случайный хвост
            labels = block["label"].to_numpy()
            pos_mask = labels > 0
            head_mask = np.zeros(len(block), dtype=bool)
            head_mask[: min(EVAL_HEAD, len(block))] = True
            rest_idx = np.flatnonzero(~pos_mask & ~head_mask)
            take = min(EVAL_TAIL_RAND, len(rest_idx))
            rand_idx = (
                rng.choice(rest_idx, size=take, replace=False)
                if take
                else np.empty(0, dtype=np.int64)
            )
            keep_idx = np.union1d(np.flatnonzero(pos_mask | head_mask), rand_idx)
            emit(block.iloc[keep_idx], "eval")
        else:
            emit(block, "holdout")
        if (i + 1) % 500 == 0:
            print(f"  обработано {i + 1}/{n_eval} eval-сессий "
                  f"({time.time() - t0:.0f} c)", flush=True)

    writer.close()
    print(f"\nСохранён: {out_path} ({time.time() - t0:.0f} c)")
    for split, st in stats.items():
        share = st["pos"] / st["rows"] if st["rows"] else 0.0
        print(f"  {split:>7}: сессий={st['sessions']}, строк={st['rows']:,}, "
              f"позитивов={st['pos']:,} ({share:.3%})")


if __name__ == "__main__":
    main()
