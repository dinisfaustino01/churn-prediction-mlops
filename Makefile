up:
	docker compose up -d

up-build:
	docker compose up -d --build

down:
	docker compose down -v

lock:
	docker compose exec airflow-webserver python -m pip freeze --exclude-editable > docker/airflow/requirements.lock
	docker compose exec mlflow python -m pip freeze > docker/mlflow/requirements.lock
	
lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest -v

install:
	uv sync

train:
	uv run python scripts/run_training.py

predict:
	uv run python scripts/batch_predict.py