import os
import sys
import shutil
import subprocess
from dotenv import load_dotenv
from pathlib import Path

def check_kaggle_setup():
    load_dotenv()
    # 1. Check if Kaggle CLI is available in the system PATH
    kaggle_cli_path = shutil.which("kaggle")
    is_cli_available = kaggle_cli_path is not None

    # 2. Check for Kaggle API credentials in environment variables
    # Kaggle officially uses KAGGLE_USERNAME and KAGGLE_KEY.
    # Some workflows also use KAGGLE_API_TOKEN.
    username = os.environ.get("KAGGLE_USERNAME")
    kaggle_key = os.environ.get("KAGGLE_KEY")
    api_token = os.environ.get("KAGGLE_API_TOKEN")

    has_key = bool(kaggle_key or api_token)
    has_username = bool(username)

    # Display Results
    print("=== Kaggle Environment Check ===")
    if is_cli_available:
        print(f"[✔] Kaggle CLI found at: {kaggle_cli_path}")
    else:
        print("[✘] Kaggle CLI not found. (Install via: pipx install kaggle)")

    if has_username and has_key:
        print("[✔] Kaggle authentication environment variables are configured.")
        if kaggle_key:
            print("    - Found: KAGGLE_KEY")
        if api_token:
            print("    - Found: KAGGLE_API_TOKEN")
    else:
        print("[✘] Kaggle authentication environment variables are missing or incomplete.")
        if not has_username:
            print("    - Missing: KAGGLE_USERNAME")
        if not has_key:
            print("    - Missing: KAGGLE_KEY (or KAGGLE_API_TOKEN)")

    return is_cli_available and (has_username and has_key)


def _load_data():
    RAW_DIR = Path("datasets/raw")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    datasets = {
        "tmdb-movies-930k": "asaniczka/tmdb-movies-dataset-2023-930k-movies",
        "the-movies-dataset": "rounakbanik/the-movies-dataset",
        "movielens-full": "grouplens/movielens-latest-full",
        "tmdb-movie-metadata": "tmdb/tmdb-movie-metadata"
    }

    if check_kaggle_setup():
        for folder, slug in datasets.items():
            dest = RAW_DIR / folder
            if dest.exists() and any(dest.iterdir()):
                print(f"  Already exists, skipping: {dest}")
                continue

            dest.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {slug}...")
            result = subprocess.run([
                "kaggle", "datasets", "download",
                "-d", slug,
                "-p", str(dest),
                "--unzip"
            ])

            if result.returncode == 0:
                print(f"  Saved to {dest}")
            else:
                print(f"  FAILED: {slug} — skipping")

        print("\nDone.")

if __name__ == '__main__':
    _load_data()