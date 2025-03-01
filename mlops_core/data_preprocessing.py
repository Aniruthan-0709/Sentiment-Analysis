import pandas as pd
import os
import logging
from collections import Counter
from imblearn.over_sampling import SMOTE

# Setup logging
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "mlops_preprocessing_pipeline.log")

# Ensure logs directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# Fix: Set UTF-8 encoding for logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),  # Enforce UTF-8 encoding
        logging.StreamHandler()
    ]
)

# File paths
RAW_DATA_PATH = "data/raw/reviews.csv"
PROCESSED_DATA_PATH = "data/processed/reviews.parquet"

def preprocess_data(input_path=RAW_DATA_PATH, output_path=PROCESSED_DATA_PATH):
    """Cleans, encodes, balances (SMOTE), and adds sentiment analysis to the dataset."""
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

        # Check class distribution
        sentiment_counts = Counter(df["review_sentiment"])
        logging.info(f"🔹 Initial Sentiment Distribution: {sentiment_counts}")

        # Handling class imbalance using SMOTE
        target_column = "review_sentiment"
        min_count = min(sentiment_counts.values())
        max_count = max(sentiment_counts.values())

        if max_count / min_count > 1.5:  # Apply SMOTE only if imbalance is significant
            logging.info("🔹 Applying SMOTE Oversampling...")
            smote = SMOTE(sampling_strategy="auto", random_state=42)
            X = df.drop(columns=[target_column])
            y = df[target_column]

            # SMOTE works only on numerical features, drop text columns before applying
            X_numeric = X.select_dtypes(include=["number"])

            X_resampled, y_resampled = smote.fit_resample(X_numeric, y)

            # Reconstruct the DataFrame
            df_resampled = pd.DataFrame(X_resampled, columns=X_numeric.columns)
            df_resampled[target_column] = y_resampled

            # Merge back non-numeric columns
            df_non_numeric = X.drop(columns=X_numeric.columns)
            df_non_numeric_resampled = df_non_numeric.iloc[:len(df_resampled)].reset_index(drop=True)
            df = pd.concat([df_resampled, df_non_numeric_resampled], axis=1)

        # Save processed data
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_parquet(output_path)

        logging.info(f"✅ Data preprocessing completed. Saved to {output_path}")
        logging.info(f"🔹 Balanced Sentiment Distribution: {Counter(df['review_sentiment'])}")
        return output_path
    except Exception as e:
        logging.error(f"❌ Error during preprocessing: {e}")
        return None

# Run the function
if __name__ == "__main__":
    preprocess_data()
