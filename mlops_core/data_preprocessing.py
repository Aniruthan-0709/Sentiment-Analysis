import pandas as pd
import os
import logging
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler

# Setup logging
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "mlops_preprocessing_pipeline.log")

# Ensure logs directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# Set UTF-8 encoding for logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# File paths
RAW_DATA_PATH = "data/raw/reviews.csv"
PROCESSED_PARQUET_PATH = "data/processed/reviews.parquet"
PROCESSED_CSV_PATH = PROCESSED_PARQUET_PATH.replace(".parquet", ".csv")

def preprocess_data(input_path=RAW_DATA_PATH, parquet_path=PROCESSED_PARQUET_PATH, csv_path=PROCESSED_CSV_PATH):
    """Cleans, scales star ratings, encodes, balances (SMOTE), and adds sentiment analysis."""
    try:
        logging.info("🔹 Loading raw data...")
        df = pd.read_csv(input_path)

        logging.info("🔹 Dropping duplicates & handling missing values...")
        df = df.drop_duplicates()
        df = df.dropna(subset=["star_rating", "review_body", "product_category"])

        # ✅ Scale `star_rating` to range 1-5 if it's outside the bounds
        if df["star_rating"].min() < 1 or df["star_rating"].max() > 5:
            logging.warning(f"⚠️ Star ratings out of range! Scaling to 1-5...")
            scaler = MinMaxScaler(feature_range=(1, 5))
            df["star_rating"] = scaler.fit_transform(df[["star_rating"]]).round().astype(int)

        logging.info("🔹 Standardizing text data...")
        df["review_body"] = df["review_body"].str.lower().str.replace(r"[^\w\s]", "", regex=True)

        logging.info("🔹 Encoding categorical columns...")
        df["product_category_encoded"] = df["product_category"].astype("category").cat.codes

        logging.info("🔹 Creating sentiment labels...")
        df["review_sentiment"] = df["star_rating"].apply(
            lambda x: "negative" if x in [1, 2] else "neutral" if x == 3 else "positive"
        )

        # Check class distribution
        sentiment_counts = Counter(df["review_sentiment"])
        logging.info(f"🔹 Initial Sentiment Distribution: {sentiment_counts}")

        # ✅ Apply SMOTE only if dataset has multiple sentiment classes and imbalance is significant
        target_column = "review_sentiment"
        min_count = min(sentiment_counts.values())
        max_count = max(sentiment_counts.values())

        if len(sentiment_counts) > 2 and max_count / min_count > 1.5:  # Apply SMOTE only if needed
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

        # ✅ Save processed data in both Parquet and CSV formats
        os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
        df.to_parquet(parquet_path)
        df.to_csv(csv_path, index=False)

        logging.info(f"✅ Data preprocessing completed. Saved to {parquet_path} & {csv_path}")
        logging.info(f"🔹 Balanced Sentiment Distribution: {Counter(df['review_sentiment'])}")
        return parquet_path, csv_path
    except Exception as e:
        logging.error(f"❌ Error during preprocessing: {e}")
        return None, None

# Run the function
if __name__ == "__main__":
    preprocess_data()
