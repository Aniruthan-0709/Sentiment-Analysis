#**Amazon Customer Sentiment Analysis - Data Pipeline**  
 **IE 7374: Data Pipeline Assignment - Group 7**  

## **Team Members**  
- Aniruthan Swaminathan Arulmurugan  
- Manikandan Mohan
- Janani Karthikeyan  
- Manivannan Senthilkumar  
- Kiran Tamilselvan  
- Aravind Anbazhagan  

---

## **Project Overview**  
Understanding customer sentiment is crucial for businesses to enhance user experience, improve product offerings, and make data-driven decisions. This project focuses on analyzing **Amazon customer reviews** to classify them into **positive, neutral, or negative sentiments** using **machine learning and natural language processing (NLP)** techniques.  

### **Key Challenges & Solutions**  
✔ **Large-scale Data Processing:** Handles **54.41GB of Amazon US Customer Reviews Dataset**.  
✔ **Data Imbalance:** Uses **SMOTE** to balance sentiment distribution.  
✔ **Automated Workflow:** **Apache Airflow** orchestrates data ingestion, validation, preprocessing, bias detection, and anomaly detection.  
✔ **Bias & Ethical Concerns:** **TensorFlow Data Validation (TFDV)** ensures fairness by monitoring data drift.  

---

## **Data Source**  
- **Dataset Name:** Amazon US Customer Reviews  
- **Format:** Tab-Separated Values (.tsv)  
- **Key Fields:** `review_id`, `product_id`, `star_rating`, `review_body`, `review_date`  
- **Source:** [Amazon US Customer Reviews on Kaggle](https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset)  

---

## **🛠Installation & Setup**  

### **Prerequisites**  
Ensure you have the following installed before running the pipeline:  
✔ **Python >= 3.9**  
✔ **Git**  
✔ **Docker & Docker-Compose** (for containerization)  
✔ **Google Cloud Platform (GCP) Account** (for cloud storage)  
✔ **Apache Airflow** (for workflow automation)  

---

### **Step-by-Step Setup**  

### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/Aniruthan-0709/Sentiment-Analysis-Tool-for-Product-Reviews.git
cd Sentiment-Analysis-Tool-for-Product-Reviews
```

### **2️⃣ Check System Requirements**  
Ensure your system meets the following criteria:  

- **OS Compatibility:** Windows, Mac, Linux  
- **Python Version:** `>=3.9`  
- **Sufficient Memory for Docker:** Minimum **8GB RAM**  
- **Verify Docker Installation:**  
  ```bash
  docker --version
  docker-compose --version
  ```

- Increase Docker’s memory allocation if needed via **Docker Desktop > Settings > Resources**  

---

### **3️⃣ Configure Environment Variables**  
Before running the pipeline, configure your **Google Cloud Storage (GCS) access**:

1. **Place your Google Cloud Service Account Key (`key.json`)** in the `config/` folder.  
2. **Update the `.env` file**:  
   ```bash
   GCP_BUCKET=mlops_dataset123
   SOURCE_BLOB=mlops_dataset123/data/raw/Sampled_11_Product_Categories.csv
   GOOGLE_APPLICATION_CREDENTIALS=config/key.json
   ```

---

### **4️⃣ Build the Docker Image**  
The entire pipeline runs inside **Docker**, managed by **Airflow DAGs**.  

```bash
docker-compose build
```

This installs **Apache Airflow** and all required dependencies inside the container.  

---

### **5️⃣ Start the Airflow Scheduler & Web Server**  
```bash
docker-compose up -d
```

This starts:  
✔ **Airflow Scheduler** (to execute DAGs)  
✔ **Airflow Web UI** (accessible at `http://localhost:8080`)  

To check if the containers are running:  
```bash
docker ps
```

---

### **6️⃣ Initialize Airflow Database & DAGs**  
```bash
docker-compose up airflow-init
```

This initializes **Airflow's metadata database** and registers **DAGs** in the scheduler.  

---

### **7️⃣ Verify Airflow DAGs**  
1. Open a web browser and visit **[http://localhost:8080](http://localhost:8080)**  
2. **Login credentials for Airflow UI:**  
   - **Username:** `admin`  
   - **Password:** `admin`  
3. Verify that the following **DAGs** are listed:  

   | **DAG Name**            | **Description**                          |
   |-------------------------|------------------------------------------|
   | `data_ingestion`        | Downloads dataset from **GCS**.          |
   | `data_preprocessing`    | Cleans, encodes, balances, and saves.    |
   | `schema_validator`      | Validates dataset schema.                |
   | `bias_detector`         | Detects bias & data drift.               |
   | `anomalies`             | Identifies anomalies & triggers alerts.  |

4. Click **"Trigger DAG"** (Play button) to start execution.  

---

### **8️⃣ Stop & Restart the Pipeline**  
To **stop** all running containers:  
```bash
docker-compose down
```

To **restart** the pipeline:  
```bash
docker-compose up -d
```

---

## **Pipeline Architecture**  

### **Workflow Stages**  
**Data Ingestion (`data_ingestion.py`)** – Fetches customer reviews from **GCS**.  
**Data Preprocessing (`data_preprocessing.py`)** – Cleans, encodes, and balances the dataset.  
**Schema Validation (`schema_validator.py`)** – Checks schema integrity using **TFDV**.  
**Bias Detection (`bias_detector.py`)** – Detects potential biases and drift.  
**Anomaly Detection (`anomalies.py`)** – Identifies unusual patterns & triggers alerts.  

### **Data Flow**  
```bash
GCS Storage → Airflow DAGs → Processed Data → Model Training
```

---

## **Tools & Technologies Used**  

### **MLOps Tools**  
✔ **GitHub Actions** – CI/CD automation  
✔ **Docker** – Containerization  
✔ **Apache Airflow** – Workflow scheduling  
✔ **Google Cloud Storage (GCS)** – Data storage  
✔ **Data Version Control (DVC)** – Dataset tracking  

### **Machine Learning & Data Processing**  
✔ **Pandas, NumPy, SciPy** – Data manipulation  
✔ **Scikit-learn, Imbalanced-learn (SMOTE)** – Data preprocessing  
✔ **TensorFlow Data Validation (TFDV)** – Schema validation & anomaly detection  

---

## **Email Notification Setup**  
DAGs send email alerts upon **successful completion or failure**.  
To enable this, configure **SMTP settings** in **docker-compose.yaml**:  

```yaml
AIRFLOW__SMTP__SMTP_HOST: smtp.gmail.com
AIRFLOW__SMTP__SMTP_USER: user@example.com
AIRFLOW__SMTP__SMTP_PASSWORD: ********
AIRFLOW__SMTP__SMTP_PORT: 587
AIRFLOW__SMTP__SMTP_MAIL_FROM: user@example.com
```

🔹 **Test Email Notification:**  
If the DAG completes successfully, an email is sent.  
If an error occurs, an **alert is triggered** to notify the team.  

---

## **Failure Handling & Alerts**  
**Schema anomalies, data drift, or pipeline failures** trigger automatic email alerts.  
If **model drift exceeds the threshold**, **automatic retraining** is triggered.  

```yaml
AIRFLOW__SCHEDULER__ENABLE_HEALTH_CHECK: 'true'
```
