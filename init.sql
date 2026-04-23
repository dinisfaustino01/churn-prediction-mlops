CREATE DATABASE mlflow_db;

CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    customer_id VARCHAR(50),
    churn_probability FLOAT,
    predicted_label INTEGER,
    prediction_timestamp TIMESTAMP,
    model_version VARCHAR(50)
);
