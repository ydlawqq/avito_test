# Пайплайн Avito-ретривала (BM25 + XGBRanker). Три команды:
#   make test    — можно ли начинать: окружение, зависимости, raw_data, скрипты
#   make train   — полное обучение модели с нуля (параметры из configs/config.yaml)
#   make answer  — итоговый ответ -> answer/answer.csv

PYTHON := .venv/bin/python
ifeq ($(wildcard $(PYTHON)),)
PYTHON := python
endif

.DEFAULT_GOAL := test
.PHONY: test train answer

test: ## проверка готовности к обучению
	$(PYTHON) scripts/test_setup.py

train: ## обучение модели с нуля (~3-4 ч)
	$(PYTHON) scripts/train_model.py

answer: ## итоговый ответ -> answer/answer.csv (~30-60 мин)
	$(PYTHON) scripts/make_answer.py
