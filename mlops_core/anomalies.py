import tensorflow_data_validation as tfdv
import pandas as pd
import os
import logging
import sys

# Setup logging with UTF-8 encoding
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "mlops_anomalies_pipeline.log")

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),  # Ensure UTF-8 encoding
        logging.StreamHandler(sys.stdout)
    ]
)

# File paths
PROCESSED_DATA_PATH = "data/processed/reviews.parquet"
SCHEMA_PATH = "validation/schema.pbtxt"
NEW_STATS_PATH = "validation/new_stats.pb"

def detect_anomalies(input_path=PROCESSED_DATA_PATH, schema_path=SCHEMA_PATH):
    """Detects data anomalies using TensorFlow Data Validation (TFDV)."""
    try:
        logging.info("🔹 Loading schema...")
        if not os.path.exists(schema_path):
            logging.error(f"❌ Schema file not found: {schema_path}")
            return None
        schema = tfdv.load_schema_text(schema_path)

        logging.info("🔹 Loading new dataset...")
        if not os.path.exists(input_path):
            logging.error(f"❌ Processed dataset not found: {input_path}")
            return None
        df = pd.read_parquet(input_path)

        logging.info("🔹 Generating statistics for new dataset...")
        new_stats = tfdv.generate_statistics_from_dataframe(df)

        os.makedirs(os.path.dirname(NEW_STATS_PATH), exist_ok=True)

        # ✅ Fix: Use the correct function `write_stats_text()`
        tfdv.write_stats_text(new_stats, NEW_STATS_PATH)

        logging.info("🔹 Running anomaly detection...")
        anomalies = tfdv.validate_statistics(new_stats, schema)

        if anomalies.anomaly_info:
            logging.warning(f"⚠️ Detected anomalies: {anomalies}")
        else:
            logging.info("✅ No anomalies found.")

        return anomalies
    except Exception as e:
        logging.error(f"❌ Error during anomaly detection: {e}")
        return None

# Run the function
if __name__ == "__main__":
    detect_anomalies()
