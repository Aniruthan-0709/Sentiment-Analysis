import tensorflow_data_validation as tfdv
import pandas as pd
import os
import logging

# Setup logging
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "mlops_bias_pipeline.log")

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

# File paths
PROCESSED_DATA_PATH = "data/processed/reviews.parquet"
SCHEMA_PATH = "validation/schema.pbtxt"
REFERENCE_STATS_PATH = "validation/reference_stats.pb"
NEW_STATS_PATH = "validation/new_stats.pb"
BIAS_REPORT_PATH = "validation/bias_report.txt"

def detect_bias(input_path=PROCESSED_DATA_PATH, schema_path=SCHEMA_PATH, reference_stats_path=REFERENCE_STATS_PATH):
    """Detects potential bias in the dataset."""
    try:
        logging.info("📊 Loading schema...")
        schema = tfdv.load_schema_text(schema_path)

        logging.info("📥 Loading processed dataset...")
        df = pd.read_parquet(input_path)

        logging.info("📊 Generating statistics for bias detection...")
        new_stats = tfdv.generate_statistics_from_dataframe(df)

        if os.path.exists(reference_stats_path):
            logging.info("🔄 Comparing with reference dataset...")
            reference_stats = tfdv.load_statistics(reference_stats_path)
            drift = tfdv.detect_dataset_drift(new_stats, reference_stats)

            # Log drift details
            with open(BIAS_REPORT_PATH, "w") as f:
                f.write(str(drift))

            logging.info(f"🚨 Bias report saved at {BIAS_REPORT_PATH}")
        else:
            logging.warning("⚠️ No reference dataset found. Bias detection is limited.")

        # Save new statistics for future comparisons
        tfdv.write_stats_text(new_stats, NEW_STATS_PATH)

        logging.info("✅ Bias detection complete.")
        return BIAS_REPORT_PATH
    except Exception as e:
        logging.error(f"❌ Error during bias detection: {e}")
        return None

# Run the function
if __name__ == "__main__":
    detect_bias()
