import tensorflow_data_validation as tfdv
import pandas as pd
import os
import logging
import sys

# Setup logging
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "mlops_anomalies_pipeline.log")

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)

PROCESSED_DATA_PATH = "data/processed/reviews.parquet"
SCHEMA_PATH = "validation/schema.pbtxt"
NEW_STATS_PATH = "validation/new_stats.tfrecord"

def detect_anomalies(input_path=PROCESSED_DATA_PATH, schema_path=SCHEMA_PATH):
    """Detects data anomalies using TensorFlow Data Validation (TFDV) and returns JSON-serializable results."""
    try:
        if not os.path.exists(schema_path):
            logging.error(f"❌ Schema file not found: {schema_path}")
            return {"error": "Schema file not found"}

        logging.info("🔹 Loading schema...")
        schema = tfdv.load_schema_text(schema_path)

        if not os.path.exists(input_path):
            logging.error(f"❌ Processed dataset not found: {input_path}")
            return {"error": "Processed dataset not found"}

        logging.info("🔹 Loading new dataset...")
        df = pd.read_parquet(input_path)

        logging.info("🔹 Generating statistics for new dataset...")
        new_stats = tfdv.generate_statistics_from_dataframe(df)

        os.makedirs(os.path.dirname(NEW_STATS_PATH), exist_ok=True)
        tfdv.write_stats_text(new_stats, NEW_STATS_PATH)

        logging.info("🔹 Running anomaly detection...")
        anomalies = tfdv.validate_statistics(new_stats, schema)

        # ✅ Convert `Anomalies` to a dictionary (JSON-serializable)
        anomalies_dict = {}
        if anomalies.anomaly_info:
            logging.warning(f"⚠️ Anomalies detected!")
            for feature, anomaly in anomalies.anomaly_info.items():
                anomalies_dict[feature] = anomaly.description
                logging.warning(f"⚠️ Feature: {feature}, Description: {anomaly.description}")
        else:
            logging.info("✅ No anomalies found.")

        return anomalies_dict  # ✅ Return a dictionary instead of `Anomalies`
    
    except Exception as e:
        logging.error(f"❌ Error during anomaly detection: {e}")
        return {"error": str(e)}

# Run the function
if __name__ == "__main__":
    detect_anomalies()
