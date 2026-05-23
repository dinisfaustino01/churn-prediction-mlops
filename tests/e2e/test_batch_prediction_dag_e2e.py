"""End-to-end test for the batch prediction DAG.

Triggers the DAG via Airflow's REST API against the running stack
and verifies it completes successfully.

Prerequisites:
    - Full Docker stack must be running (docker compose up -d)
    - A champion model must be registered in MLflow under the "champion" alias
    - A reference snapshot must exist for drift detection
    - A valid CSV batch file must be present in data/incoming/
"""

import time

import pytest
import requests


def _wait_for_dag(session, base_url, dag_id, dag_run_id, timeout=300, interval=10):

    deadline = time.time() + timeout

    while time.time() < deadline:
        response = session.get(f"{base_url}/dags/{dag_id}/dagRuns/{dag_run_id}")
        state = response.json()["state"]

        if state == "success":
            return "success"
        if state == "failed":
            return "failed"

        time.sleep(interval)

    return "timeout"


@pytest.mark.e2e
def test_batch_prediction_dag_e2e():

    session = requests.Session()
    session.auth = ("admin", "admin")
    base_url = "http://airflow-webserver:8080/api/v1"

    dag_id = "batch_prediction_pipeline"

    response = session.patch(f"{base_url}/dags/{dag_id}", json={"is_paused": False})
    assert response.status_code == 200, f"Failed to unpause DAG: {response.text}"

    response = session.post(f"{base_url}/dags/{dag_id}/dagRuns", json={"conf": {}})
    assert response.status_code == 200, f"Failed to trigger DAG: {response.text}"
    dag_run_id = response.json()["dag_run_id"]

    result = _wait_for_dag(
        session=session,
        base_url=base_url,
        dag_id=dag_id,
        dag_run_id=dag_run_id,
        timeout=300,
        interval=10,
    )

    assert result == "success"
