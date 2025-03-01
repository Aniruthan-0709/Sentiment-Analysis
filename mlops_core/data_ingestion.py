import os
import gcsfs
import pandas as pd
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Fetch environment variables
BUCKET_NAME = os.getenv("GCP_BUCKET")
FILE_PATH = os.getenv("SOURCE_BLOB")
SERVICE_ACCOUNT_KEY = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "config/key.json")
LOCAL_SAVE_PATH = "data/raw/reviews.csv"

# Check if env variables are loading
logging.info(f"🔹 GCP_BUCKET: {BUCKET_NAME}")
logging.info(f"🔹 SOURCE_BLOB: {FILE_PATH}")
logging.info(f"🔹 GOOGLE_APPLICATION_CREDENTIALS: {SERVICE_ACCOUNT_KEY}")

# Ensure Google credentials are set
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_KEY

def download_data():
    """Downloads dataset from GCP bucket using service account credentials."""
    try:
        logging.info("🔹 Connecting to GCS...")
        fs = gcsfs.GCSFileSystem(token=SERVICE_ACCOUNT_KEY)
        
        logging.info(f"🔹 Downloading {FILE_PATH} from {BUCKET_NAME}...")
        with fs.open(FILE_PATH) as f:
            df = pd.read_csv(f)

        os.makedirs(os.path.dirname(LOCAL_SAVE_PATH), exist_ok=True)
        df.to_csv(LOCAL_SAVE_PATH, index=False)

        logging.info(f"✅ Dataset downloaded successfully: {LOCAL_SAVE_PATH}")
        return LOCAL_SAVE_PATH
    except Exception as e:
        logging.error(f"❌ Error downloading dataset: {e}")
        return None

# Run the function
if __name__ == "__main__":
    download_data()
