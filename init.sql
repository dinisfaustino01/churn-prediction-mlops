CREATE DATABASE mlflow_db;
CREATE DATABASE airflow_db;

CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    customer_id VARCHAR(50) NOT NULL,
    churn_probability FLOAT NOT NULL,
    predicted_label INTEGER NOT NULL,
    prediction_timestamp TIMESTAMP NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    input_features_hash VARCHAR(50) NOT NULL,
    batch_filename VARCHAR(255) NOT NULL
);

CREATE TABLE IF NOT EXISTS drift_reports (
    id SERIAL PRIMARY KEY,
    batch_filename VARCHAR(255) NOT NULL,
    dataset_drift_detected BOOLEAN NOT NULL,
    num_drifted_features INTEGER NOT NULL,
    share_drifted_features FLOAT NOT NULL,
    report_path VARCHAR(512) NOT NULL,
    report_timestamp TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS data_quality_reports (
    id SERIAL PRIMARY KEY,
    batch_filename VARCHAR(255) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    check_timestamp TIMESTAMP NOT NULL,
    all_passed BOOLEAN NOT NULL,
    total_checks INTEGER NOT NULL,
    passed_checks INTEGER NOT NULL,
    failed_checks INTEGER NOT NULL,
    warnings INTEGER NOT NULL,
    check_details JSONB NOT NULL
);

CREATE INDEX idx_predictions_customer_id ON predictions(customer_id);
CREATE INDEX idx_predictions_timestamp ON predictions(prediction_timestamp);
CREATE INDEX idx_drift_reports_batch_filename ON drift_reports(batch_filename);
CREATE INDEX idx_data_quality_reports_batch_filename ON data_quality_reports(batch_filename);
