from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
from datetime import datetime, timedelta
import os

# DAG Configuration
DVC_REPO_PATH = os.getenv("DVC_REPO_PATH", "/opt/airflow/repo")  # Dynamically set DVC repo path
EMAIL_RECIPIENT = "your-email@example.com"

DEFAULT_ARGS = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2025, 2, 28),
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# DAG Definition
dag = DAG(
    "mlops_pipeline",
    default_args=DEFAULT_ARGS,
    description="MLOps Pipeline with DVC-triggered execution",
    schedule_interval=None,  # Manual Trigger + File Change Trigger
    catchup=False,
)

# Function to check for new data
def check_for_new_data():
    os.chdir(DVC_REPO_PATH)
    status = os.popen("dvc status").read()
    if "data/raw/reviews.csv" in status:
        print("✅ New data detected! Proceeding with pipeline execution.")
    else:
        raise ValueError("❌ No new data detected. Skipping pipeline.")

# Task 1: Check for New Data in DVC
check_data = PythonOperator(
    task_id="check_new_data",
    python_callable=check_for_new_data,
    dag=dag,
)

# Task 2: Pull latest data from DVC
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

# Task 9: Send Email Notification
send_email = EmailOperator(
    task_id="send_email",
    to=EMAIL_RECIPIENT,
    subject="MLOps Pipeline Completed ✅",
    html_content="<h3>Your MLOps pipeline has successfully completed!</h3>",
    dag=dag,
)

# **Task Dependencies**
check_data >> dvc_pull >> data_ingestion >> data_preprocessing
data_preprocessing >> schema_validation >> anomaly_detection >> bias_detection >> dvc_commit_push >> send_email
