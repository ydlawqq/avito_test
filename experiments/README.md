# experiments

Исследовательские / одноразовые прогоны, не входящие в основной пайплайн
(`scripts/` — только воспроизводимые шаги решения).

## recall_at_k_bm25.py

Recall@k (k = 50, 250, 500, 1000, 3000, 5000, 10000) для **чистого BM25**
(без бустов категории/локации, без dense/hybrid) на **локальной валидации** —
том же наборе запросов, что использует `scripts/evaluate.py`.

Читает ровно два файла, `raw_data/` не нужен:

| файл | что даёт |
|---|---|
| `artifacts/validation.parquet` | 2452 сессии из `train.parquet`: `query_id`, признаки запроса, `relevant_items` |
| `artifacts/items_processed.parquet` | корпус 189k объявлений benchmark_items (леммы, посчитанные `scripts/prepare_data.py`) |

* ретривер — `BM25Retriever` (`rank_bm25.BM25Okapi`) с параметрами `cfg["bm25"]`
  (`title_weight=3`, `use_params`, `use_description`, `k1`);
  `apply_metadata_boosts` не вызывается → чистый BM25-скор;
* метрика — `Recall@k = |Top-k ∩ Relevant| / |Relevant|`, среднее по 2452 сессиям;
  top-10000 достаётся один раз на запрос, recall считается кумулятивно;
* перед чтением файлы проверяются на magic-байты `PAR1` (`check_parquet`) — при
  обрезанном/недокопированном parquet выводится понятная ошибка вместо
  `pyarrow ... <Buffer>`.

```bash
python experiments/recall_at_k_bm25.py
python experiments/recall_at_k_bm25.py --out artifacts/recall_at_k_bm25.csv
python experiments/recall_at_k_bm25.py --size 200        # первые 200 сессий валидации
python experiments/recall_at_k_bm25.py --validation /path/to/other_validation.parquet
```

Лог удобно писать в `artifacts/` (`*.log` в .gitignore):

```bash
python experiments/recall_at_k_bm25.py > artifacts/recall_at_k_bm25.log 2>&1
```

### Почему не «весь трейн»

`validation.parquet` — это первые 2452 сессии из 26 556 сессий трейна
(детерминированный порядок по md5-хешу ключа запроса, `size` из `config.yaml`).
Релевантность для остальных сессий существует только в 490-МБ
`raw_data/train.parquet`, который нужен лишь там, где строится сама валидация
(`scripts/prepare_data.py`). Поэтому здесь используется тот же срез, что и в
`evaluate.py` — зато цифры сравнимы с его прогонами.

Существующий код пайплайна скрипт не меняет — использует готовые функции `src/*`.


### Итоги экспериментов.


* `recall_at_k_bm.py` :

== Чистый BM25: Recall@k на локальной валидации ==
       k   Recall@k
      50     0.2978
     250     0.5783
     500     0.7206
    1000     0.8185
    3000     0.9234
    5000     0.9458
   10000     0.9668


## bm25_pool.py + dense_rerank_10k.py — «BM25 → пул 10k → dense»

Эксперимент: НЕ индексируем весь корпус dense-моделью, а сначала отбираем
чистым BM25 top-10k на запрос (recall@10k = 0.9668) и ранжируем dense-моделью
только внутри этого пула. Кодируются только объявления, попавшие хотя бы в один
пул валидационных запросов (объединение пулов), с кэшем эмбеддингов.

### Этап A — bm25_pool.py (CPU)

Для каждой сессии валидации сохраняет пул чистого BM25 (без бустов):
`(query_id, item_id, bm25_score)` → `artifacts/bm25_pool.parquet` (zstd,
~24.5M строк, ~200 МБ). Печатает контрольную таблицу Recall@k — она должна
совпадать с результатами `recall_at_k_bm25.py`.

```bash
python experiments/bm25_pool.py                # все 2452 сессий, пул 10000
python experiments/bm25_pool.py --size 100      # быстрая проверка
```

### Этап B — dense_rerank_10k.py (GPU)

Запускается на GPU-машине. Нужны: код проекта (`experiments/`, `solution/`,
`scripts/build_dense_index.py`, `configs/`), `artifacts/validation.parquet`,
`artifacts/items_processed.parquet`, `artifacts/bm25_pool.parquet`.

* чанкование — то же, что в `scripts/build_dense_index.py` (title+params +
  окна описания, модель и параметры из `cfg["dense"]`: BAAI/bge-m3,
  chunk_words/overlap, max_chunks_per_item=3);
* кодируются только item'ы из объединения пулов; эмбеддинги — float16,
  batch 256, возобновляемый кэш `artifacts/dense/pool_shards/shard_*.npy`
  (прервали — перезапустили, продолжится с места остановки);
* на каждый запрос — локальный faiss `IndexFlatIP` только по чанкам его пула
  (~30k векторов), чанки item'а схлопываются по max-скору;
* метрики Recall@k, k = 50/250/500/1000, для вариантов:
  `bm25` (контроль), `bm25+loc` (буст локации score×(1+2.0)), `dense`,
  `rrf(w)` = RRF(bm25, dense) и `rrf+loc(w)` = RRF(bm25+loc, dense)
  при w ∈ {0.5, 1, 2, 4} (сетка настраивается `--dense-weights`);
* результат — таблица в stdout + CSV `artifacts/dense_rerank_10k.csv`.

```bash
python experiments/dense_rerank_10k.py --size 100   # пилот (кодирует меньше)
python experiments/dense_rerank_10k.py               # все сессии пула
```

Память: ~3–4 ГБ RAM (пул 24.5M строк в памяти), кэш эмбеддингов ~1 ГБ на диск;
повторные прогоны (другие веса RRF, другие k) кэш не трогают и идут быстро.

### Результаты этапа A (полный прогон, 2452 сессии, 12 мин CPU)

Контрольная таблица Recall@k чистого BM25 совпала с `recall_at_k_bm25.py`
побитово — пул построен корректно:

| k | 50 | 250 | 500 | 1000 | 10000 |
|---|---|---|---|---|---|
| Recall@k | 0.2978 | 0.5783 | 0.7206 | 0.8185 | 0.9668 |

Файл `artifacts/bm25_pool.parquet`: 21.3M строк, 290 МБ (zstd), 2451 сессия
(у одной сессии нет документов с BM25-скором > 0), дубликатов нет, порядок
по убыванию скора соблюдён. Средний размер пула 8680 (< 10000, т.к. у части
запросов меньше документов с положительным скором).

**Важная находка**: объединение пулов всех 2452 сессий покрывает 189 210 из
189 212 объявлений корпуса — на ПОЛНОЙ валидации «кодировать только пул»
экономии не даёт, кодировать придётся почти весь корпус. Экономия есть на
пилотах (`--size 100`) и главное — на этапе поиска: запрос сравнивается только
со ~26k чанков своего пула, а не с 560k всего корпуса. Кэш эмбеддингов
(`pool_shards/`) при этом переиспользуется между прогонами и сетками весов.
Если кодирование всего корпуса неприемлемо по времени — уменьшите `--pool`
## solution/xgb_rerank/ (вынесено из experiments) — XGBoost-реранкер поверх «BM25 с бустами → топ-1000»

Продакшн-пайплайн, под который строится модель:

```text
чистый BM25 -> сырой пул top-5000 (bm25.boost_pool)
            -> мягкие бусты score * (1 + loc_boost * loc_match)
            -> переранжированный ТОП-1000          <- вход реранкера
            -> XGBRanker (признаки пары запрос x объявление)
            -> финальный топ-50
```

### Headroom (по готовому artifacts/bm25_pool.parquet, 2452 сессии)

| k (переранжированный пул) | Recall@k |
|---|---|
| 50 (бейзлайн bm25+loc) | 0.7874 |
| 100 | 0.8358 |
| 250 | 0.8820 |
| 500 | 0.9067 |
| 1000 | **0.9262** |

Т.е. у реранкера на топ-1000 потолок Recall@50 ≈ 0.926 против 0.787 у
текущего бустнутого BM25 (+14 п.п.). 12.5% релевантных попадают в
бустнутый топ-1000 только благодаря бусту локации — их ранги модель и
должна выучивать.

### Признаки (solution/src/rerank/xgb_features.py)

У ЗАПРОСА доступны ТОЛЬКО `search_*` признаки (как в benchmark_queries):
`search_query`, `search_location_id`, `search_is_delivery_search`,
`search_infm_params_text`, `search_category` (99.98% = 114, delivery = 1 в
0.002% строк, фильтры пустые в 33% строк). Сторона объявления — из корпуса
(benchmark_items / items_processed). Всего 30 числовых + 2 категориальных
(microcat_id, search_loc, enable_categorical):

* retrieval-сигналы: `bm25_score`, `bm25_boosted`, `bm25_rank_raw`,
  `bm25_rank` (входной порядок топа-1000), `rank_pct`;
* метадата: `loc_match`, `cat_match`;
* лексика: пересечения/покрытия лемм запроса с заголовком/params/описанием,
  подстрочные фичи по сырому заголовку (`title_substr`, `title_startswith`);
* длины текстов и запроса, `is_delivery`;
* априорика объявления: `price_log`, `has_price`, `rating`, `reviews_log`,
  `phone_hidden`, `message_forbidden`, `microcat_freq`, `loc_freq`.

Один и тот же код фич используется на этапе датасета, обучения и инференса
(`scripts/make_answer.py --method xgb`) — расхождение train/inference исключено.

### Этап 1 — build_dataset.py (CPU, ~10–15 мин на 2452 сессии)

Для каждой сессии воспроизводит продакшн-вход: сырой BM25-топ-5000 → бусты →
топ-1000 → пары с label (пользователь выбирал объявление по запросу) и
фичами. Сплит по сессиям в md5-порядке: 70% train / 30% eval (eval-сессии не
пересекаются с validation.parquet). `--train-from raw_data/train.parquet`
берёт ВСЕ ~26.5k сессий трейна (больше данных, ~2 ч CPU); `--neg-frac 0.3`
сэмплирует негативы глубже 50-го места в train-части.

```bash
python solution/xgb_rerank/build_dataset.py > artifacts/xgb_rerank/build.log 2>&1
python solution/xgb_rerank/build_dataset.py --train-from raw_data/train.parquet \
    > artifacts/xgb_rerank/build_full.log 2>&1
```

### Этап 2 — train_xgb_reranker.py (CPU/GPU)

XGBRanker, objective rank:ndcg; early stopping по КАСТОМНОЙ set-метрике
Recall@50 (feval, имя recall@50-max — суффикс `-max` включает максимизацию;
signature per-group `(y_true, y_score)`, т.к. sklearn-интерфейс ранкера
оборачивает callable-метрики ltr_metric_decorator'ом по группам),
ndcg@50 печатается для сравнения. Train-группы без позитивов отбрасываются
(в них лямбда-градиенты нулевые). `--n-jobs` по умолчанию None: значение -1
ломает ltr-декоратор метрик (ThreadPoolExecutor(max_workers=-1)).
ВНИМАНИЕ: НЕ собирайте датасет с --neg-frac 0 — тогда в train-части
остаются только ~50 головных строк сессии и модель не видит позитивов
на местах 51–1000 (главный слой для выучивания). Рекомендуется
`--neg-frac 0.3` или полный пул.
Печатает настоящий Recall@50 на eval-сессиях: бейзлайн `bm25+loc@50`, прямой
`xgb@50`, гибриды `xgbR->loc@50` (топ-R по XGB → переранжирование бустнутым
BM25) и `xgb->rrf<w>@50` (RRF рангов). Сохраняет `model.json` +
`cat_categories.json`. `--xgb-model model.json` — дообучить поверх готовой
модели (fit xgb_model=...).

```bash
python solution/xgb_rerank/train_xgb_reranker.py > artifacts/xgb_rerank/train.log 2>&1
# дообучение существующей модели:
python solution/xgb_rerank/train_xgb_reranker.py \
    --xgb-model artifacts/xgb_rerank/model.json --eta 0.01
```

### Этап 3 — инференс

```bash
python scripts/make_answer.py --method xgb --limit 100   # пилот
python scripts/make_answer.py --method xgb               # answer.csv
```

### Этап 4 — оценка реранкера (Recall@50)

```bash
# быстрый прогон на части валидации
python scripts/evaluate.py --method xgb --limit 300

# Честная оценка без утечки: holdout-сессии датасета (не участвовали ни в
# обучении, ни в early stopping). Атрибуты запросов подтягиваются из train.parquet.
python scripts/evaluate.py --method xgb \
    --dataset artifacts/xgb_rerank/dataset_resampled.parquet --split holdout
```

#### Recall@k при k != 50

```bash
# таблица Recall@50/100/150 (реранкер отдаёт max(k)=150 кандидатов из пула 1000)
python scripts/evaluate.py --method xgb --k 50,100,150 \
    --dataset artifacts/xgb_rerank/dataset_resampled.parquet --split holdout

# на валидации (быстрее, но с утечкой train-сессий)
python scripts/evaluate.py --method xgb --k 50,100,150 --limit 300
```

Печатается таблица: `k | xgb | bm25+loc | прирост`. Реранкер возвращает `max(k)`
кандидатов из переранжированного пула (но не больше `depth`), бейзлайн берётся из
головы того же пула — сравнение честное. Ориентир (holdout, 100 сессий):
`k=50: 0.878 vs 0.851 (+2.7 п.п.)`, `k=100: 0.910 vs 0.910 (паритет)`,
`k=150: 0.930 vs 0.940 (-1.0 п.п.)` — выигрыш реранкера сосредоточен в топ-50,
а на 100/150 голова бустнутого BM25 уже почти исчерпывает recall.

Внимание про утечку: `validation.parquet`-сессии входят в train-часть датасетов
`dataset.parquet`/`dataset_resampled.parquet`, поэтому оценка без `--dataset`
оптимистична. Печатается Recall@50 реранкера и бейзлайна `bm25+loc@50` на тех же
запросах (голова того же пула), плюс прирост.

`evaluate.py --method xgb` и `make_answer.py --method xgb` вызывают ОДИН и тот же
код инференса (`solution/src/rerank/inference.py::rerank_xgb`), поэтому метрика
оценки соответствует формируемому ответу.

Важно (исправлено): в `predict_xgb` топ-50 доставался по позициям внутри пула
(`retriever.doc_ids[i]`), а не по индексам корпуса. Из-за этого ответы брались
из первых 1000 позиций корпуса и скор на бенчмарке падал до ~0.004. Теперь
маппинг корректный — `retriever.doc_ids[corpus_idx[order]]`, и результат
инференса совпадает с локальными метриками (`xgb@50` = 0.898 на holdout).
Проверка-регресс: пересечение топ-50 `xgb` и `bm25+loc` на одних запросах
должно быть ~0.6-1.0, а не 0.

Скорость инференса та же, что у метода bm25 (BM25-скоринг всего корпуса на
запрос), плюс предсказание XGB на 1000 строк — копейки.

### Smoke

`dataset_smoke.parquet` / `model_smoke.json` — прогон на 3 сессиях
(--size 3, --iterations 50) только для проверки кода; в answer.csv их не
использовать.

## Слой XGBoost-реранка (BM25 + бусты -> топ-1000 -> XGBRanker -> топ-50)

Реранкер поверх BM25 с бустами локации/категории. Пайплайн:

```
BM25 + бусты локации/категории
    -> переранжированный топ-1000 (cfg.xgb_rerank.depth)
    -> XGBRanker
    -> финальный топ-50
```

XGBoost работает на ~30 табличных признаках пары (пересечения лемм, BM25-скоры,
метаданные) и переранжирует BM25-пул глубины `depth` (по умолчанию 1000).

### Код
- `solution/src/rerank/inference.py` — `rerank_xgb`: загрузка модели, сбор признаков,
  инференс BM25+бусты -> топ-`depth` -> XGBRanker -> топ-50.
- `solution/xgb_rerank/` — обучение модели и подбор глубины (папка вынесена из `experiments/`).

### Оценка
```bash
# честная оценка без утечки: holdout-сессии (не были ни в train, ни в early stopping)
python scripts/evaluate.py --method xgb \
    --dataset artifacts/xgb_rerank/dataset_resampled.parquet --split holdout

# таблица Recall@50/100/150
python scripts/evaluate.py --method xgb --k 50,100,150 \
    --dataset artifacts/xgb_rerank/dataset_resampled.parquet --split holdout

# ответ
python scripts/make_answer.py --method xgb          # answer.csv
python scripts/make_answer.py --method xgb --limit 100   # пилот
```

### Smoke
`dataset_smoke.parquet` / `model_smoke.json` — прогон на 3 сессиях
(--size 3, --iterations 50) только для проверки кода; в answer.csv их не
использовать.

### Флаги
`--model` (model.json XGBRanker), `--categories` (cat_categories.json),
`--depth` (глубина переранжированного пула), `--dataset` (датасет для честной
оценки), `--split` (holdout/eval/train), `--k` (k для Recall@k через запятую).
Значения по умолчанию — в `cfg.xgb_rerank` (`configs/config.yaml`); явный флаг
перекрывает конфиг.


(например 2000: recall@2000 = 0.90, объединение пулов заметно уже) или
оценивайте на подвыборке сессий `--size`.
