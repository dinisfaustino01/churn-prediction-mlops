"""Unit tests for DAG structure validation.

Verifies that both DAGs load without import errors, contain the expected
task IDs, and wire the correct upstream dependencies.
"""


def test_batch_prediction_dag_structure(dag_bag):

    batch_prediction_dag = dag_bag.get_dag("batch_prediction_pipeline")

    assert dag_bag.import_errors == {}
    assert batch_prediction_dag is not None

    validate_incoming_data = batch_prediction_dag.get_task("validate_incoming_data")
    detect_drift = batch_prediction_dag.get_task("detect_drift")
    run_predictions = batch_prediction_dag.get_task("run_predictions")
    check_data_quality = batch_prediction_dag.get_task("check_data_quality")
    archive_processed_data = batch_prediction_dag.get_task("archive_processed_data")

    assert set(batch_prediction_dag.task_ids) == {
        "validate_incoming_data",
        "detect_drift",
        "run_predictions",
        "check_data_quality",
        "archive_processed_data",
    }

    assert validate_incoming_data.upstream_task_ids == set()
    assert detect_drift.upstream_task_ids == {"validate_incoming_data"}
    assert run_predictions.upstream_task_ids == {"detect_drift"}
    assert check_data_quality.upstream_task_ids == {"run_predictions"}
    assert archive_processed_data.upstream_task_ids == {"check_data_quality"}


def test_retraining_dag_structure(dag_bag):

    retraining_dag = dag_bag.get_dag("retraining_pipeline")

    assert dag_bag.import_errors == {}
    assert retraining_dag is not None

    train_candidate_task = retraining_dag.get_task("train_candidate_task")
    compare_and_register_task = retraining_dag.get_task("compare_and_register_task")
    notify_retraining_outcome_task = retraining_dag.get_task(
        "notify_retraining_outcome_task"
    )

    assert set(retraining_dag.task_ids) == {
        "train_candidate_task",
        "compare_and_register_task",
        "notify_retraining_outcome_task",
    }

    assert train_candidate_task.upstream_task_ids == set()
    assert compare_and_register_task.upstream_task_ids == {"train_candidate_task"}
    assert notify_retraining_outcome_task.upstream_task_ids == {
        "compare_and_register_task"
    }
