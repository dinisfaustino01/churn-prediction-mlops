import pytest
from airflow.models import DagBag

from churn_prediction import PROJECT_ROOT


@pytest.fixture
def dag_bag():
    dag_folder_path = PROJECT_ROOT / "dags"
    dag_bag = DagBag(dag_folder=dag_folder_path, include_examples=False)
    return dag_bag
