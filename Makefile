.PHONY: setup data features train backtest api dashboard test lint all clean

setup:
	pip install -r requirements.txt -r requirements-dev.txt && pip install -e .

data:
	python -m alphabench.cli ingest --universe config/universe_in.yaml
	python -m alphabench.cli validate

features:
	python -m alphabench.cli build-features

train:
	python -m alphabench.cli train --model lightgbm --horizon 1

backtest:
	python -m alphabench.cli backtest --model lightgbm --horizon 1

holdout:
	python -m alphabench.cli evaluate-holdout --model lightgbm --horizon 1

api:
	uvicorn alphabench.api.main:app --reload --port 8000

dashboard:
	streamlit run src/alphabench/dashboard/app.py

test:
	pytest tests/ -v

lint:
	ruff check src tests && ruff format --check src tests && mypy src

all: data features train backtest

clean:
	rm -rf data/interim/* data/processed/* models/* reports/figures/*
