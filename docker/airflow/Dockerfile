FROM apache/airflow:2.10.0

USER airflow

COPY docker/airflow/requirements.lock .
RUN pip install --no-deps -r requirements.lock

COPY pyproject.toml .
COPY churn_prediction/ ./churn_prediction
RUN pip install --no-deps -e .
