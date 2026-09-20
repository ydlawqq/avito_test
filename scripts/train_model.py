"""Полное переобучение XGBoost-реранкера одной командой (все параметры — из
раздела [train] в configs/config.yaml).

Запуск из корня проекта:
    python scripts/train_model.py                 # пропускает готовые артефакты
    python scripts/train_model.py --force         # пересобрать всё с нуля
    python scripts/train_model.py --steps dataset,train

Цепочка (воспроизводит v2 -> model_retrain.json):
    1. prepare    scripts/prepare_data.py           (если артефактов ещё нет)
    2. pair_stats solution/xgb_rerank/pair_stats.py
                   -> train.outputs.pair_stats (9 таблиц парных статистик)
    3. dataset    solution/xgb_rerank/build_dataset.py --train-from
                   raw_data/train.parquet (все ~26.5k сессий, пулы top-1000,
                   train-негативы отбрасываются: neg_frac=0) ~2.5 ч
    4. augment    solution/xgb_rerank/augment_dataset.py — векторная добивка
                   v2-фич (гео, парные статистики, jaccard); обязателен:
                   build_dataset строит только базовые фичи
    5. resample   solution/xgb_rerank/resample_dataset.py — перенос
                   eval-сессий с полными пулами в train (глубокие негативы)
    6. train      solution/xgb_rerank/train_xgb_reranker.py
                   (rank:ndcg, depth=6, eta=0.05, early stopping 100)
                   -> train.outputs.model / .categories

После проверки holdout-метрики укажите пути в xgb_rerank.model/.categories
конфига (production model_v2.json эта команда не затирает).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PAIR_STATS_TABLES = ["item_geo", "loc_centroid", "loc_pairs", "loc_microcat",
                     "microcat_pos", "qtext_cnt", "qtext_microcat",
                     "qtext_item", "item_pos"]


def sh(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if r.returncode != 0:
        sys.exit(f"\nШаг упал (rc={r.returncode}) через {time.time() - t0:.0f} c. "
                 f"Останов.")
    print(f"[шаг завершён за {time.time() - t0:.0f} c]")


def pair_stats_complete(ps_dir: Path) -> bool:
    return ps_dir.exists() and all(
        (ps_dir / f"{t}.parquet").exists() for t in PAIR_STATS_TABLES)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "config.yaml"))
    ap.add_argument("--force", action="store_true",
                    help="пересобрать все артефакты, даже если они уже есть")
    ap.add_argument("--steps", default="",
                    help="запустить только перечисленные шаги через запятую "
                         "(prepare,pair_stats,dataset,augment,resample,train)")
    args = ap.parse_args()

    import yaml
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    tr = cfg["train"]
    out = tr["outputs"]
    paths = {k: PROJECT_ROOT / v for k, v in out.items()}
    only = [s.strip() for s in args.steps.split(",") if s.strip()]

    def want(step: str) -> bool:
        return not only or step in only

    force = args.force
    py = sys.executable
    exp = str(PROJECT_ROOT / "solution" / "xgb_rerank")
    print("=" * 72)
    print("Полное переобучение реранкера (параметры из configs/config.yaml)")
    print("=" * 72)

    # 1. prepare -----------------------------------------------------------
    items = PROJECT_ROOT / cfg["paths"]["artifacts_dir"] / "items_processed.parquet"
    validation = PROJECT_ROOT / cfg["paths"]["artifacts_dir"] / "validation.parquet"
    if want("prepare"):
        if items.exists() and validation.exists() and not force:
            print("\n[prepare] артефакты уже есть — пропуск (--force пересоберёт)")
        else:
            sh([py, "scripts/prepare_data.py", "--config", args.config])

    # 2. pair_stats --------------------------------------------------------
    if want("pair_stats"):
        if pair_stats_complete(paths["pair_stats"]) and not force:
            print("\n[pair_stats] таблицы уже есть — пропуск")
        else:
            sh([py, f"{exp}/pair_stats.py", "--out", str(paths["pair_stats"])])

    # 3. dataset -----------------------------------------------------------
    if want("dataset"):
        if paths["dataset"].exists() and not force:
            print(f"\n[dataset] {paths['dataset']} уже есть — пропуск")
        else:
            sh([py, f"{exp}/build_dataset.py", "--config", args.config,
                "--train-from", str(PROJECT_ROOT / cfg["paths"]["raw_data_dir"]
                                    / "train.parquet"),
                "--depth", str(int(tr["build_depth"])),
                "--train-frac", str(float(tr["train_frac"])),
                "--neg-frac", str(float(tr["neg_frac"])),
                "--seed", str(int(tr["seed"])),
                "--out", str(paths["dataset"])])

    # 4. augment -----------------------------------------------------------
    if want("augment"):
        if paths["dataset_v2"].exists() and not force:
            print(f"\n[augment] {paths['dataset_v2']} уже есть — пропуск")
        else:
            sh([py, f"{exp}/augment_dataset.py", "--config", args.config,
                "--src", str(paths["dataset"]),
                "--out", str(paths["dataset_v2"]),
                "--stats-dir", str(paths["pair_stats"]),
                "--chunk-rows", str(int(tr["augment_chunk_rows"]))])

    # 5. resample ----------------------------------------------------------
    if want("resample"):
        if paths["dataset_resampled"].exists() and not force:
            print(f"\n[resample] {paths['dataset_resampled']} уже есть — пропуск")
        else:
            sh([py, f"{exp}/resample_dataset.py",
                "--dataset", str(paths["dataset_v2"]),
                "--out", str(paths["dataset_resampled"])])

    # 6. train -------------------------------------------------------------
    if want("train"):
        x = tr["xgb"]
        cmd = [py, f"{exp}/train_xgb_reranker.py",
               "--dataset", str(paths["dataset_resampled"]),
               "--iterations", str(int(x["iterations"])),
               "--max-depth", str(int(x["max_depth"])),
               "--eta", str(float(x["eta"])),
               "--early-stopping", str(int(x["early_stopping"])),
               "--subsample", str(float(x["subsample"])),
               "--colsample", str(float(x["colsample"])),
               "--objective", str(x["objective"]),
               "--seed", str(int(tr["seed"])),
               "--model-out", str(paths["model"]),
               "--categories-out", str(paths["categories"])]
        if x.get("n_jobs"):
            cmd += ["--n-jobs", str(int(x["n_jobs"]))]
        sh(cmd)

    print("\n" + "=" * 72)
    print(f"Готово. Модель: {paths['model']}")
    print(f"Категории:      {paths['categories']}")
    print("Проверьте holdout-метрику в логе обучения, затем для перехода на "
          "новую модель поправьте в configs/config.yaml:")
    print(f"  xgb_rerank.model:      {out['model']}")
    print(f"  xgb_rerank.categories: {out['categories']}")
    print("и получите ответ: python scripts/make_answer.py")


if __name__ == "__main__":
    main()
