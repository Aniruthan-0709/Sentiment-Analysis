import tensorflow_data_validation as tfdv
import tensorflow as tf
import pandas as pd
import os
import logging
import sys
from tensorflow_metadata.proto.v0 import statistics_pb2  # ✅ Correct import

# Setup logging
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "mlops_bias_pipeline.log")

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
REFERENCE_STATS_PATH = "validation/reference_stats.tfrecord"
NEW_STATS_PATH = "validation/new_stats.tfrecord"
BIAS_REPORT_PATH = "validation/bias_report.txt"

def load_statistics_from_tfrecord(path):
    """Load dataset statistics from TFRecord binary file."""
    dataset = tf.data.TFRecordDataset([path])
    for record in dataset:
        stats = statistics_pb2.DatasetFeatureStatisticsList()  # ✅ Corrected
        stats.ParseFromString(record.numpy())
        return stats
    return None

def save_statistics_as_tfrecord(stats, path):
    """Save dataset statistics as TFRecord binary file."""
    with tf.io.TFRecordWriter(path) as writer:
        writer.write(stats.SerializeToString())

def detect_bias(input_path=PROCESSED_DATA_PATH, schema_path=SCHEMA_PATH, reference_stats_path=REFERENCE_STATS_PATH):
    """Detects potential bias in the dataset by comparing it with a reference dataset."""
    try:
        logging.info("📊 Loading schema...")
        if not os.path.exists(schema_path):
            logging.error(f"❌ Schema file not found: {schema_path}")
            return None
        schema = tfdv.load_schema_text(schema_path)

        logging.info("📥 Loading processed dataset...")
        if not os.path.exists(input_path):
            logging.error(f"❌ Processed dataset not found: {input_path}")
            return None
        df = pd.read_parquet(input_path)

        logging.info("📊 Generating statistics for bias detection...")
        new_stats = tfdv.generate_statistics_from_dataframe(df)

        os.makedirs(os.path.dirname(BIAS_REPORT_PATH), exist_ok=True)

        if os.path.exists(reference_stats_path):
            logging.info("🔄 Comparing with reference dataset...")
            reference_stats = load_statistics_from_tfrecord(reference_stats_path)

            # Compute drift manually
            drift_results = []
            for feature in new_stats.datasets[0].features:
                feature_name = feature.path.step[0]
                ref_feature = next(
                    (f for f in reference_stats.datasets[0].features if f.path.step[0] == feature_name), None
                )

                if ref_feature and feature.HasField("num_stats") and ref_feature.HasField("num_stats"):
                    old_mean = ref_feature.num_stats.mean
                    new_mean = feature.num_stats.mean
                    percent_drift = ((new_mean - old_mean) / old_mean) * 100 if old_mean != 0 else 0

                    drift_results.append(f"{feature_name}: Old Mean = {old_mean:.3f}, New Mean = {new_mean:.3f}, Drift = {percent_drift:.2f}%")

            # Save bias report with structured format
            with open(BIAS_REPORT_PATH, "w", encoding="utf-8") as f:
                f.write("📊 Bias Detection Report\n")
                f.write("="*40 + "\n")
                for entry in drift_results:
                    f.write(entry + "\n")

            logging.info(f"🚨 Bias report saved at {BIAS_REPORT_PATH}")
        else:
            logging.warning("⚠️ No reference dataset found. Saving current stats as reference for future bias detection.")
            save_statistics_as_tfrecord(new_stats, reference_stats_path)

        save_statistics_as_tfrecord(new_stats, NEW_STATS_PATH)

        logging.info("✅ Bias detection complete.")
        return BIAS_REPORT_PATH
    except Exception as e:
        logging.error(f"❌ Error during bias detection: {e}")
        return None

# Run the function
if __name__ == "__main__":
    detect_bias()
