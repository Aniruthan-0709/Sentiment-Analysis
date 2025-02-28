import pandas as pd
import os
import logging

# Setup logging
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "mlops_preprocessing_pipeline.log")

# Ensure logs directory exists
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
RAW_DATA_PATH = "data/raw/reviews.csv"
PROCESSED_DATA_PATH = "data/processed/reviews.parquet"

def preprocess_data(input_path=RAW_DATA_PATH, output_path=PROCESSED_DATA_PATH):
    """Cleans, encodes, and adds sentiment analysis to the dataset."""
    try:
        logging.info("🔹 Loading raw data...")
        df = pd.read_csv(input_path)

        logging.info("🔹 Dropping duplicates & handling missing values...")
        df = df.drop_duplicates()
        df = df.dropna(subset=["star_rating", "review_body", "product_category"])

        logging.info("🔹 Standardizing text data...")
        df["review_body"] = df["review_body"].str.lower().str.replace(r"[^\w\s]", "", regex=True)

        logging.info("🔹 Encoding categorical columns...")
        df["product_category_encoded"] = df["product_category"].astype("category").cat.codes
        df["star_rating"] = df["star_rating"].astype(int)

        logging.info("🔹 Creating sentiment labels...")
        df["review_sentiment"] = df["star_rating"].apply(
            lambda x: "negative" if x in [1, 2] else "neutral" if x == 3 else "positive"
        )

        # Save processed data
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_parquet(output_path)

        logging.info(f"✅ Data preprocessing completed. Saved to {output_path}")
        return output_path
    except Exception as e:
        logging.error(f"❌ Error during preprocessing: {e}")
        return None

# Run the function
if __name__ == "__main__":
    preprocess_data()
