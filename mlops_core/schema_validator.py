import tensorflow_data_validation as tfdv
import tensorflow as tf
import pandas as pd
import os
import logging
import sys

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
REFERENCE_STATS_PATH = "validation/reference_stats.tfrecord"  # ✅ Change extension to .tfrecord

def save_statistics_as_tfrecord(stats, path):
    """Save dataset statistics as TFRecord binary file."""
    with tf.io.TFRecordWriter(path) as writer:
        writer.write(stats.SerializeToString())

def validate_schema(input_path=PROCESSED_DATA_PATH, schema_path=SCHEMA_PATH, stats_path=REFERENCE_STATS_PATH):
    """Validates dataset schema using TensorFlow Data Validation (TFDV)."""
    try:
        logging.info("🔹 Loading processed data for schema validation...")
        df = pd.read_parquet(input_path)

        logging.info("🔹 Generating statistics for schema inference...")
        stats = tfdv.generate_statistics_from_dataframe(df)

        logging.info("🔹 Inferring schema...")
        schema = tfdv.infer_schema(stats)

        os.makedirs(os.path.dirname(schema_path), exist_ok=True)

        # ✅ Save the inferred schema
        tfdv.write_schema_text(schema, schema_path)
        logging.info(f"✅ Schema validation completed. Schema saved at {schema_path}")

        # ✅ Save reference stats as TFRecord instead of text
        save_statistics_as_tfrecord(stats, stats_path)
        logging.info(f"✅ Reference statistics saved at {stats_path}")

        return schema_path, stats_path
    except Exception as e:
        logging.error(f"❌ Error during schema validation: {e}")
        return None, None

# Run the function
if __name__ == "__main__":
    validate_schema()
