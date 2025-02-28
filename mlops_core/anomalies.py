import tensorflow_data_validation as tfdv
import pandas as pd
import os
import logging

# Setup logging
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "mlops_anomalies_pipeline.log")

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),  # Log to file
        logging.StreamHandler()         # Log to console
    ]
)

# File paths
PROCESSED_DATA_PATH = "data/processed/reviews.parquet"
SCHEMA_PATH = "validation/schema.pbtxt"
REFERENCE_STATS_PATH = "validation/reference_stats.pb"
NEW_STATS_PATH = "validation/new_stats.pb"

def detect_anomalies(input_path=PROCESSED_DATA_PATH, schema_path=SCHEMA_PATH):
    """Detects data anomalies using TensorFlow Data Validation (TFDV)."""
    try:
        logging.info("🔹 Loading schema...")
        schema = tfdv.load_schema_text(schema_path)

        logging.info("🔹 Loading new dataset...")
        df = pd.read_parquet(input_path)

        logging.info("🔹 Generating statistics for new dataset...")
        new_stats = tfdv.generate_statistics_from_dataframe(df)

        # Save the new statistics
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
