"""Проверка готовности окружения к воспроизведению решения.

Запуск из корня проекта:
    python scripts/test_setup.py           # быстрые проверки (~1 мин)
    python scripts/test_setup.py --smoke   # + мини-прогон всей цепочки обучения
                                           #   (60 сессий, ~5-10 мин)

Без --smoke проверяется:
  1. версия Python и наличие всех зависимостей (requirements.txt);
  2. структура configs/config.yaml (разделы paths/bm25/xgb_rerank/train);
  3. наличие и читаемость raw_data/*.parquet с нужными колонками;
  4. наличие всех скриптов пайплайна (scripts/, solution/xgb_rerank/);
  5. импортируемость модулей solution и паритет фич с model_v2.json;
  6. запись в artifacts/.

Отсутствие производных артефактов (items_processed, pair_stats, модель) —
не ошибка: их создаст `make train`. Ошибка — только то, что нельзя
починить запуском обучения (зависимости, исходные данные, скрипты).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "solution"))

OK, WARN, FAIL = "OK  ", "WARN", "FAIL"
problems: list[str] = []


def report(status: str, msg: str) -> None:
    print(f"  [{status}] {msg}")
    if status == FAIL:
        problems.append(msg)


def load_config() -> dict:
    import yaml
    with open(PROJECT_ROOT / "configs" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------- 1. python/dep
def check_environment() -> None:
    print("\n1) Python и зависимости")
    v = sys.version_info
    report(OK if v >= (3, 10) else FAIL,
           f"Python {v.major}.{v.minor}.{v.micro}")

    required = ["yaml", "numpy", "pandas", "pyarrow", "xgboost",
                "rank_bm25", "pymorphy3", "tqdm"]
    for mod in required:
        try:
            m = __import__(mod)
            report(OK, f"{mod} {getattr(m, '__version__', '?')}")
        except ImportError as e:
            report(FAIL, f"{mod}: не импортируется ({e}) — pip install -r requirements.txt")


# ---------------------------------------------------------------- 2. config
def check_config(cfg: dict) -> None:
    print("\n2) configs/config.yaml")
    for section, keys in {
        "paths": ["raw_data_dir", "artifacts_dir", "answer_dir"],
        "bm25": ["k1", "b", "boost_pool", "location_boost", "title_weight"],
        "xgb_rerank": ["depth", "model", "categories", "pair_stats"],
        "train": ["seed", "train_frac", "neg_frac", "build_depth", "xgb", "outputs"],
    }.items():
        if section not in cfg:
            report(FAIL, f"нет раздела [{section}]")
            continue
        missing = [k for k in keys if k not in cfg[section]]
        report(OK if not missing else FAIL,
               f"[{section}]" + (f" — не хватает ключей: {missing}" if missing else ""))


# ---------------------------------------------------------------- 3. raw_data
def check_raw_data() -> None:
    print("\n3) raw_data (исходные данные)")
    from src.evaluation.validation import QUERY_KEY_COLUMNS
    specs = {
        "train.parquet": QUERY_KEY_COLUMNS + ["item_id"],
        "benchmark_queries.parquet": ["query_id"] + QUERY_KEY_COLUMNS,
        "benchmark_items.parquet": ["item_id", "item_location_id",
                                    "item_microcat_id", "item_latitude",
                                    "item_longitude"],
    }
    import pandas as pd
    for name, cols in specs.items():
        p = PROJECT_ROOT / "raw_data" / name
        if not p.exists() or p.stat().st_size < 1024:
            report(FAIL, f"{name}: отсутствует или пуст (загрузите исходные данные)")
            continue
        try:
            df = pd.read_parquet(p)
            missing = [c for c in cols if c not in df.columns]
            report(OK if not missing else FAIL,
                   f"{name}: {len(df):,} строк, колонки" +
                   (f" — отсутствуют: {missing}" if missing else " — все на месте"))
        except Exception as e:
            report(FAIL, f"{name}: не читается ({e})")


# ---------------------------------------------------------------- 4. скрипты
def check_scripts() -> None:
    print("\n4) Скрипты пайплайна")
    files = [
        "scripts/prepare_data.py", "scripts/make_answer.py", "scripts/evaluate.py",
        "solution/xgb_rerank/pair_stats.py",
        "solution/xgb_rerank/build_dataset.py",
        "solution/xgb_rerank/augment_dataset.py",
        "solution/xgb_rerank/resample_dataset.py",
        "solution/xgb_rerank/train_xgb_reranker.py",
        "solution/src/rerank/xgb_features.py", "solution/src/rerank/inference.py",
        "solution/src/retrievers/retrieval.py", "solution/src/pipeline/common.py",
    ]
    for f in files:
        report(OK if (PROJECT_ROOT / f).exists() else FAIL, f)


# ---------------------------------------------------------------- 5. импорты/фичи
def check_model_parity(cfg: dict) -> None:
    print("\n5) Модули solution и паритет фич с моделью")
    try:
        from src.rerank.xgb_features import CAT_FEATURES, FEATURE_NAMES, NUMERIC_FEATURES
        report(OK, f"FEATURE_NAMES: {len(FEATURE_NAMES)} "
                   f"({len(NUMERIC_FEATURES)} числовых + {len(CAT_FEATURES)} категор.)")
    except Exception as e:
        report(FAIL, f"импорт src.rerank.xgb_features: {e}")
        return

    model_path = PROJECT_ROOT / cfg["xgb_rerank"]["model"]
    if not model_path.exists():
        report(WARN, f"модель {model_path} отсутствует — обучение её создаст, "
                     f"но инференс make_answer.py не запустится")
        return
    import xgboost as xgb
    bst = xgb.Booster()
    bst.load_model(str(model_path))
    fnames = list(bst.feature_names or [])
    report(OK, f"модель {model_path.name}: {len(fnames)} фич, "
               f"{bst.num_boosted_rounds()} деревьев")
    extra = [f for f in fnames if f not in FEATURE_NAMES]
    absent = [f for f in FEATURE_NAMES if f not in fnames]
    report(OK if not extra and not absent else FAIL,
           "паритет фич код<->модель" +
           (f"; модель ждёт отсутствующие в коде: {extra}" if extra else "") +
           (f"; код строит лишние для модели: {absent}" if absent else ""))


# ---------------------------------------------------------------- 6. артефакты
def check_artifacts(cfg: dict) -> None:
    print("\n6) Артефакты")
    artifacts = PROJECT_ROOT / cfg["paths"]["artifacts_dir"]
    for name in ("items_processed.parquet", "validation.parquet"):
        p = artifacts / name
        if p.exists():
            report(OK, f"artifacts/{name} ({p.stat().st_size / 1e6:.0f} МБ)")
        else:
            report(WARN, f"artifacts/{name} отсутствует — создастся на шаге 1 "
                         f"команды make train")

    ps_dir = PROJECT_ROOT / cfg["train"]["outputs"]["pair_stats"]
    tables = ["item_geo", "loc_centroid", "loc_pairs", "loc_microcat",
              "microcat_pos", "qtext_cnt", "qtext_microcat", "qtext_item",
              "item_pos"]
    if ps_dir.exists():
        import pandas as pd
        bad = []
        for t in tables:
            f = ps_dir / f"{t}.parquet"
            if not f.exists():
                bad.append(t)
                continue
            try:
                pd.read_parquet(f)
            except Exception:
                bad.append(t)
        report(OK if not bad else FAIL,
               f"pair_stats: {len(tables) - len(bad)}/{len(tables)} таблиц" +
               (f"; битые/отсутствуют: {bad}" if bad else ""))
    else:
        report(WARN, f"{ps_dir} отсутствует — создаётся шагом pair_stats "
                     f"в train_model.py")

    for key in ("model", "categories"):
        p = PROJECT_ROOT / cfg["xgb_rerank"][key]
        report(OK if p.exists() else WARN,
               f"xgb_rerank.{key}: {p}" + ("" if p.exists() else " — отсутствует"))

    try:
        artifacts.mkdir(parents=True, exist_ok=True)
        (artifacts / ".write_test").touch()
        (artifacts / ".write_test").unlink()
        report(OK, f"запись в {artifacts}/ разрешена")
    except OSError as e:
        report(FAIL, f"нет записи в {artifacts}: {e}")


# ---------------------------------------------------------------- smoke
def run(cmd: list[str]) -> bool:
    print(f"\n$ {' '.join(cmd[1:])}")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=PROJECT_ROOT)
    ok = r.returncode == 0
    report(OK if ok else FAIL, f"rc={r.returncode}, {time.time() - t0:.0f} c")
    return ok


def run_smoke(cfg: dict) -> None:
    print("\n7) SMOKE: мини-прогон всей цепочки обучения "
          "(60 сессий, мелкие пулы, 15 деревьев)")
    if problems:
        print("  пропущен: есть FAIL на предыдущих шагах")
        return
    py = sys.executable
    smoke = PROJECT_ROOT / cfg["paths"]["artifacts_dir"] / "test_smoke"
    smoke.mkdir(parents=True, exist_ok=True)
    cfg_path = str(PROJECT_ROOT / "configs" / "config.yaml")
    ps = str(PROJECT_ROOT / cfg["train"]["outputs"]["pair_stats"])

    chain = [
        ("build_dataset",
         [py, "solution/xgb_rerank/build_dataset.py", "--config", cfg_path,
          "--size", "60", "--depth", "200", "--pool", "2000", "--neg-frac", "0",
          "--out", str(smoke / "dataset.parquet")]),
        ("augment_dataset",
         [py, "solution/xgb_rerank/augment_dataset.py", "--config", cfg_path,
          "--src", str(smoke / "dataset.parquet"),
          "--out", str(smoke / "dataset_v2.parquet"), "--stats-dir", ps]),
        ("resample_dataset",
         [py, "solution/xgb_rerank/resample_dataset.py",
          "--dataset", str(smoke / "dataset_v2.parquet"),
          "--out", str(smoke / "dataset_resampled.parquet"),
          "--limit-eval-sessions", "18"]),
        ("train_xgb_reranker",
         [py, "solution/xgb_rerank/train_xgb_reranker.py",
          "--dataset", str(smoke / "dataset_resampled.parquet"),
          "--iterations", "15", "--max-depth", "4",
          "--model-out", str(smoke / "model_smoke.json"),
          "--categories-out", str(smoke / "cats_smoke.json")]),
        ("make_answer",
         [py, "scripts/make_answer.py", "--method", "xgb", "--limit", "3",
          "--depth", "200", "--model", str(smoke / "model_smoke.json"),
          "--categories", str(smoke / "cats_smoke.json"),
          "--out", str(smoke / "answer_smoke.csv")]),
    ]
    for name, cmd in chain:
        print(f"\n  --- шаг: {name}")
        if not run(cmd):
            print(f"\n  SMOKE FAILED на шаге {name}; файлы: {smoke}")
            problems.append(f"smoke: шаг {name}")
            return
    n_rows = len((smoke / "answer_smoke.csv").read_text().strip().splitlines()) - 1
    report(OK if n_rows == 3 else FAIL, f"answer_smoke.csv: {n_rows}/3 строк")
    print(f"\n  SMOKE OK — вся цепочка (датасет -> фичи -> resample -> обучение "
          f"-> инференс) работает. Файлы: {smoke}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true",
                    help="мини-прогон всей цепочки обучения (~5-10 мин)")
    args = ap.parse_args()

    print("=" * 72)
    print("Проверка готовности к воспроизведению решения")
    print("=" * 72)
    check_environment()
    cfg = load_config()
    check_config(cfg)
    check_raw_data()
    check_scripts()
    check_model_parity(cfg)
    check_artifacts(cfg)
    if args.smoke:
        run_smoke(cfg)

    print("\n" + "=" * 72)
    if problems:
        print(f"ИТОГ: {len(problems)} проблема(ы):")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("ИТОГ: обучение можно запускать:  make train   # ~3-4 ч")
    print("затем итоговый ответ:            make answer  # -> answer/answer.csv")
    sys.exit(0)


if __name__ == "__main__":
    main()
