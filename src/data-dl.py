import os
import requests
import zipfile
import pandas as pd

# CONFIG ----
file_url = "https://github.com/datamisc/ess-10/raw/main/data.zip"
data_dir = "data/raw/"
zip_path = os.path.join(data_dir, "data.zip")
csv_filename = "ESS10.csv"     
csv_path = os.path.join(data_dir, csv_filename)

# Ensure directory exists
os.makedirs(data_dir, exist_ok=True)

# FUNCTION to Download data if not available
def ensure_csv_available():
    if os.path.exists(csv_path):
        print(f"✅ CSV already exists: {csv_path}")
        return

    print("❗ CSV not found. Downloading ZIP...")

    # Download ZIP
    response = requests.get(file_url)
    response.raise_for_status()
    with open(zip_path, "wb") as f:
        f.write(response.content)
    print(f"📦 ZIP downloaded to: {zip_path}")

    # Extract ZIP
    print("📂 Extracting ZIP...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_dir)

    print(f"✅ Files extracted to: {data_dir}")

    # Confirm extraction
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV '{csv_filename}' not found inside ZIP. "
            f"Check ZIP contents in {data_dir}."
        )

# ---- MAIN ----
ensure_csv_available()

print("📄 Loading CSV...")
df = pd.read_csv(csv_path)

print("✅ Data loaded. First rows:")
print(df.head())
