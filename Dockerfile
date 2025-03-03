# Use the official Apache Airflow image with Python 3.9 as the base
FROM apache/airflow:2.6.3-python3.9

# Install system dependencies (if needed)
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Switch back to airflow user
USER airflow

# Set environment variables for Airflow
ENV AIRFLOW_HOME=/opt/airflow

# Set PYTHONPATH to include the project directory
ENV PYTHONPATH="${AIRFLOW_HOME}:${AIRFLOW_HOME}/mlops_core"

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip

# Install TensorFlow and related dependencies
RUN pip install --no-cache-dir tensorflow==2.12.0
RUN pip install --no-cache-dir tensorflow-data-validation==1.13.0

# Install other dependencies
RUN pip install --no-cache-dir pandas==1.5.3
RUN pip install --no-cache-dir "pyarrow>=6,<7"
RUN pip install --no-cache-dir scikit-learn==1.0.2  # Ensure scikit-learn is installed
RUN pip install --no-cache-dir imbalanced-learn==0.10.1
RUN pip install --no-cache-dir apache-airflow-providers-google==10.4.0
RUN pip install --no-cache-dir python-dotenv
RUN pip install --no-cache-dir tqdm
RUN pip install --no-cache-dir gcsfs  # Add gcsfs to the list of dependencies
RUN pip install --no-cache-dir protobuf==3.20.*
# Copy over the project files
COPY dags /opt/airflow/dags/
COPY config /opt/airflow/config/
COPY mlops_core /opt/airflow/mlops_core/
