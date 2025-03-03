from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from airflow.utils.email import send_email_smtp
from airflow.operators.email import EmailOperator


# Importing functions from scripts
from mlops_core.data_ingestion import download_data
from mlops_core.data_preprocessing import preprocess_data
from mlops_core.schema_validator import validate_schema
from mlops_core.anomalies import detect_anomalies
from mlops_core.bias_detector import detect_bias

# Default arguments for the DAG
default_args = {
    'owner': 'aniruthanhpe',
    'depends_on_past': False,
    'email': ['aniruthanhpe@gmail.com'],  # Email notifications
    'email_on_failure': True,
    'email_on_success': True,
    'email_on_retry': False,
    'retries': 1,
}

# Define the DAG
with DAG(
    dag_id='mlops_pipeline',
    default_args=default_args,
    description='MLOps pipeline with schema validation, anomaly detection, and bias detection',
    schedule_interval=None,  # Manual trigger only
    start_date=days_ago(1),
    catchup=False,
) as dag:

    # Task 1: Data Ingestion
    ingestion_task = PythonOperator(
        task_id='data_ingestion',
        python_callable=download_data,
    )

    # Task 2: Data Preprocessing
    preprocessing_task = PythonOperator(
        task_id='data_preprocessing',
        python_callable=preprocess_data,
    )

    # Task 3: Schema Validation
    schema_validation_task = PythonOperator(
        task_id='schema_validation',
        python_callable=validate_schema,
    )

    # Task 4: Anomaly Detection
    anomaly_detection_task = PythonOperator(
        task_id='anomaly_detection',
        python_callable=detect_anomalies,
    )

    # Task 5: Bias Detection
    bias_detection_task = PythonOperator(
        task_id='bias_detection',
        python_callable=detect_bias,
    )

    success_email = EmailOperator(
        task_id='send_success_email',
        to='aniruthanhpe@gmail.com',
        subject='✅ MLOps Pipeline Completed Successfully',
        html_content='<p>Your MLOps pipeline finished successfully! 🎉</p>',
        dag=dag
    )
    # Define dependencies (linear pipeline)
    ingestion_task >> preprocessing_task >> schema_validation_task >> anomaly_detection_task >> bias_detection_task>>success_email
