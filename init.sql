CREATE DATABASE mlflow_db;
CREATE DATABASE airflow_db;

CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    customer_id VARCHAR(50),
    churn_probability FLOAT,
    predicted_label INTEGER,
    prediction_timestamp TIMESTAMP,
    model_version VARCHAR(50),
    input_features_hash VARCHAR(50),
    batch_filename VARCHAR(255)
);

CREATE TABLE IF NOT EXISTS drift_reports (
    id SERIAL PRIMARY KEY,
    batch_filename VARCHAR(255),
    dataset_drift_detected BOOLEAN,
    num_drifted_features INTEGER,
    share_drifted_features FLOAT,
    report_path VARCHAR(512),
    report_timestamp TIMESTAMP
);

CREATE INDEX idx_predictions_customer_id ON predictions(customer_id);
CREATE INDEX idx_predictions_timestamp ON predictions(prediction_timestamp);