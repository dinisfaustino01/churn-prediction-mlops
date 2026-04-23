FROM ghcr.io/mlflow/mlflow:v2.10.0

USER root
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

RUN pip install psycopg2-binary
