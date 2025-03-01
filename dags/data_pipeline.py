from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.email import EmailOperator
from datetime import datetime, timedelta
import os

# ✅ Load environment variables
DVC_REPO_PATH = "/opt/airflow"  # Root directory where the repo is stored
EMAIL_RECIPIENT = "aniruthanhpe@gmail.com"

DEFAULT_ARGS = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2025, 2, 28),
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "mlops_pipeline",
    default_args=DEFAULT_ARGS,
    description="Runs Data Ingestion first, then processing, validation, anomalies/bias detection, and DVC tracking",
    schedule_interval=None,  # Runs manually when triggered
    catchup=False,
)

# ✅ Task 1: Run Data Ingestion (Replaces reviews.csv)
data_ingestion = BashOperator(
    task_id="data_ingestion",
    bash_command=f"cd {DVC_REPO_PATH} && python mlops_core/data_ingestion.py",
    dag=dag,
)

# ✅ Task 2: Run Data Preprocessing
data_preprocessing = BashOperator(
    task_id="data_preprocessing",
    bash_command=f"cd {DVC_REPO_PATH} && python mlops_core/data_preprocessing.py",
    dag=dag,
)

# ✅ Task 3: Run Schema Validation
schema_validation = BashOperator(
    task_id="schema_validation",
    bash_command=f"cd {DVC_REPO_PATH} && python mlops_core/schema_validator.py",
    dag=dag,
)

# ✅ Task 4: Run Anomaly Detection
anomaly_detection = BashOperator(
    task_id="anomaly_detection",
    bash_command=f"cd {DVC_REPO_PATH} && python mlops_core/anomalies.py",
    dag=dag,
)

# ✅ Task 5: Run Bias Detection
bias_detection = BashOperator(
    task_id="bias_detection",
    bash_command=f"cd {DVC_REPO_PATH} && python mlops_core/bias_detector.py",
    dag=dag,
)

# ✅ Task 6: Version and Push Processed Data using DVC (Detects changes in reviews.csv)
dvc_commit_push = BashOperator(
    task_id="dvc_commit_push",
    bash_command=f"""
    cd {DVC_REPO_PATH} &&
    dvc add data/processed/ validation/ &&
    git add data.dvc validation.dvc &&
    git commit -m 'Updated processed dataset and validation results' &&
    git push origin master &&
    dvc push
    """,
    dag=dag,
)

# ✅ Task 7: Send Email Notification
send_email = EmailOperator(
    task_id="send_email",
    to=EMAIL_RECIPIENT,
    subject="MLOps Pipeline Completed ✅",
    html_content="<h3>Your MLOps pipeline has successfully completed!</h3>",
    dag=dag,
)

# **Task Dependencies (Runs Ingestion First, Then Other Steps)**
data_ingestion >> data_preprocessing
data_preprocessing >> schema_validation >> anomaly_detection >> bias_detection >> dvc_commit_push >> send_email
