# Failure Mode Matrix

---

## Data Load

**File missing**

- Detection: `FileNotFoundError` caught at task start
- Recovery: Retry 2x with backoff, then alert + halt DAG

**Schema mismatch**

- Detection: Pydantic validation on load
- Recovery: Fail loud, no retry — data error, not transient

**File empty (0 rows)**

- Detection: Row count check after load
- Recovery: Halt DAG, alert — nothing to process

**Duplicate rows**

- Detection: Row count vs unique ID count check
- Recovery: Log warning, deduplicate, continue

---

## Preprocessing

**Unseen categorical value**

- Detection: sklearn raises on transform
- Recovery: Catch, log warning, impute with most frequent

**All nulls in a column**

- Detection: Null ratio check before fit
- Recovery: Halt pipeline, alert — column is unusable

**Pipeline artifact missing**

- Detection: `FileNotFoundError` on MLflow load
- Recovery: Halt DAG, alert — cannot score without pipeline

---

## Training

**MLflow tracking server unreachable**

- Detection: Try/except on `mlflow.start_run()`
- Recovery: Retry 3x, then fail loud — run is not reproducible without logging

**Early stopping never triggers**

- Detection: Max rounds reached check
- Recovery: Log warning, continue — model may be overfit, flag in run metadata

**Model worse than baseline**

- Detection: AUC comparison post-eval
- Recovery: Log, do not register — keep current Production model

---

## Model Registration

**MLflow registry unreachable**

- Detection: Try/except on `mlflow.register_model()`
- Recovery: Retry 3x, then halt — model exists but is unversioned

**No Production model exists**

- Detection: `MlflowException` on stage fetch
- Recovery: Halt DAG, alert — nothing to load for inference

---

## Batch Prediction

**Production model missing**

- Detection: `MlflowException` on model load
- Recovery: Halt DAG, alert — do not write partial predictions

**Prediction output contains NaNs**

- Detection: Post-prediction null check
- Recovery: Halt write, alert — corrupted output is worse than no output

**Postgres write fails**

- Detection: SQLAlchemy exception on insert
- Recovery: Retry 3x with backoff, then alert + halt

**Prediction distribution collapses (all 0s or all 1s)**

- Detection: Distribution check post-scoring
- Recovery: Write predictions, alert — model may be degraded

---

## Drift Detection

**Evidently report fails to generate**

- Detection: Try/except on report run
- Recovery: Log error, skip drift check, alert — do not block predictions

**Drift detected**

- Detection: Drift score exceeds threshold
- Recovery: Trigger retraining DAG, alert

**Reference dataset missing**

- Detection: `FileNotFoundError` on reference load
- Recovery: Halt drift check, alert — cannot compare without baseline

---

## Alerting

**Webhook unreachable**

- Detection: HTTP error on alert send
- Recovery: Retry 2x, then write to local log — alerting failure must not crash the DAG
