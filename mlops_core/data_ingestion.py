import os
import gcsfs
import pandas as pd
import logging
from dotenv import load_dotenv
from tqdm import tqdm  # Import tqdm for progress bar

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Fetch environment variables
BUCKET_NAME = os.getenv("GCP_BUCKET")
FILE_PATH = os.getenv("SOURCE_BLOB")
SERVICE_ACCOUNT_KEY = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "config/key.json")
LOCAL_SAVE_PATH = "data/raw/reviews.csv"

# Ensure Google credentials are set
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_KEY

def download_data(chunk_size=8192):  # Define chunk size for downloading
    """Downloads dataset from GCP bucket using service account credentials."""
    try:
        # Check if the file already exists
        if os.path.exists(LOCAL_SAVE_PATH):
            logging.info(f"✅ File already exists: {LOCAL_SAVE_PATH}")
            return LOCAL_SAVE_PATH

        logging.info("🔹 Connecting to GCS...")
        fs = gcsfs.GCSFileSystem(token=SERVICE_ACCOUNT_KEY)
        
        logging.info(f"🔹 Downloading {FILE_PATH} from {BUCKET_NAME}...")
        
        # Get the file size for the progress bar
        file_size = fs.size(FILE_PATH)
        
        with fs.open(FILE_PATH, 'rb') as f:
            os.makedirs(os.path.dirname(LOCAL_SAVE_PATH), exist_ok=True)
            
            with open(LOCAL_SAVE_PATH, 'wb') as local_file:
                with tqdm(total=file_size, unit='B', unit_scale=True, desc=FILE_PATH) as pbar:
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        local_file.write(chunk)
                        pbar.update(len(chunk))

        logging.info(f"✅ Dataset downloaded successfully: {LOCAL_SAVE_PATH}")
        return LOCAL_SAVE_PATH
    except Exception as e:
        logging.error(f"❌ Error downloading dataset: {e}")
        return None

# Run the function
if __name__ == "__main__":
    download_data()
