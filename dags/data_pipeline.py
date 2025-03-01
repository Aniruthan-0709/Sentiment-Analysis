from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from datetime import datetime, timedelta
import os

# DAG Configuration
DVC_REPO_PATH = "/d/NEU/Spring 2025/MLops/repo-root"  # Update to your repo path
DEFAULT_ARGS = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2025, 2, 28),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# DAG Definition
dag = DAG(
    "mlops_pipeline",
    default_args=DEFAULT_ARGS,
    description="MLops Pipeline with automatic DVC-triggered execution",
    schedule_interval=None,  # Manual Trigger + File Change Trigger
    catchup=False,
)

# Function to check for new data
def check_for_new_data(**kwargs):
    os.chdir(DVC_REPO_PATH)
    status = os.popen("dvc status").read()

    # Detect changes in raw data or schema
    new_data_detected = "data/raw/reviews.csv" in status or "validation/schema.pbtxt" in status

    if new_data_detected:
        print("✅ New data detected! Proceeding with pipeline execution.")
        return True  # Continue DAG Execution
    else:
        print("⚠️ No new data detected. Skipping pipeline.")
        return False  # ShortCircuit (Skip Execution)

# Task 1: Check for New Data in DVC
check_data = ShortCircuitOperator(
    task_id="check_new_data",
    python_callable=check_for_new_data,
    provide_context=True,
    dag=dag,
)

# Task 2: Pull latest data from DVC (if changes exist)
dvc_pull = BashOperator(
    task_id="dvc_pull",
    bash_command=f"cd {DVC_REPO_PATH} && dvc pull",
    dag=dag,
)

# Task 3: Run Data Ingestion
data_ingestion = BashOperator(
    task_id="data_ingestion",
    bash_command=f"cd {DVC_REPO_PATH} && python mlops_core/data_ingestion.py",
    dag=dag,
)

# Task 4: Run Preprocessing
data_preprocessing = BashOperator(
    task_id="data_preprocessing",
    bash_command=f"cd {DVC_REPO_PATH} && python mlops_core/data_preprocessing.py",
    dag=dag,
)

# Task 5: Schema Validation
schema_validation = BashOperator(
    task_id="schema_validation",
    bash_command=f"cd {DVC_REPO_PATH} && python mlops_core/schema_validator.py",
    dag=dag,
)

# Task 6: Anomaly Detection
anomaly_detection = BashOperator(
    task_id="anomaly_detection",
    bash_command=f"cd {DVC_REPO_PATH} && python mlops_core/anomalies.py",
    dag=dag,
)

# Task 7: Bias Detection
bias_detection = BashOperator(
    task_id="bias_detection",
    bash_command=f"cd {DVC_REPO_PATH} && python mlops_core/bias_detector.py",
    dag=dag,
)

# Task 8: Version & Push Processed Data + Validation to DVC
dvc_commit_push = BashOperator(
    task_id="dvc_commit_push",
    bash_command=f"""
    cd {DVC_REPO_PATH} &&
    dvc add data/ validation/ &&
    git add data.dvc validation.dvc &&
    git commit -m 'Updated dataset and validation results' &&
    git push origin master &&
    dvc push
    """,
    dag=dag,
)

# **Task Dependencies**
check_data >> dvc_pull >> data_ingestion >> data_preprocessing
data_preprocessing >> schema_validation >> anomaly_detection >> bias_detection >> dvc_commit_push
