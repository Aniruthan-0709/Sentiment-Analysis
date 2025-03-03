import tensorflow_data_validation as tfdv
import tensorflow as tf
import pandas as pd
import os
import logging
import sys
from tensorflow_data_validation.utils import schema_util

# Setup logging
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "mlops_schema_pipeline.log")

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)

# File paths
PROCESSED_DATA_PATH = "data/processed/reviews.parquet"
SCHEMA_PATH = "validation/schema.pbtxt"
REFERENCE_STATS_PATH = "validation/reference_stats.tfrecord"  # ✅ Stores reference statistics

def save_statistics_as_tfrecord(stats, path):
    """Save dataset statistics as TFRecord binary file."""
    with tf.io.TFRecordWriter(path) as writer:
        writer.write(stats.SerializeToString())

def validate_schema(input_path=PROCESSED_DATA_PATH, schema_path=SCHEMA_PATH, stats_path=REFERENCE_STATS_PATH):
    """Validates dataset schema using TensorFlow Data Validation (TFDV)."""
    try:
        logging.info("🔹 Loading processed data for schema validation...")
        df = pd.read_parquet(input_path)
        
        logging.info("🔹 Generating statistics for schema validation...")
        stats = tfdv.generate_statistics_from_dataframe(df)

        # ✅ Check if schema already exists
        if os.path.exists(schema_path):
            logging.info("🔹 Existing schema found. Validating against it...")
            schema = tfdv.load_schema_text(schema_path)
            anomalies = tfdv.validate_statistics(stats, schema)

            # ✅ Log schema anomalies
            if schema_util.anomalies_present(anomalies):
                logging.warning(f"⚠️ Schema drift detected! {anomalies}")
                
                # Raise error if critical schema change is found
                for feature in anomalies.anomaly_info:
                    if "missing" in anomalies.anomaly_info[feature].description.lower():
                        raise ValueError(f"🚨 Critical schema change! Column '{feature}' is missing.")
                else:
                    logging.info("🔹 Schema anomalies detected but not critical.")
            else:
                logging.info("✅ No schema anomalies detected.")

        else:
            logging.info("🔹 No existing schema found. Inferring and saving schema...")
            schema = tfdv.infer_schema(stats)
            os.makedirs(os.path.dirname(schema_path), exist_ok=True)

            tfdv.write_schema_text(schema, schema_path)
            logging.info(f"✅ Schema saved at {schema_path}")

            # ✅ Save reference statistics
            save_statistics_as_tfrecord(stats, stats_path)
            logging.info(f"✅ Reference statistics saved at {stats_path}")

        return schema_path, stats_path
    except Exception as e:
        logging.error(f"❌ Error during schema validation: {e}")
        return None, None

# Run the function
if __name__ == "__main__":
    validate_schema()
