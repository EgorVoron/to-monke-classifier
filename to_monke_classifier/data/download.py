import logging
import subprocess


def download_dvc_data():
    try:
        logging.info("Downloading with dvc pull...")
        subprocess.run(["dvc", "pull"], check=True)
        print("Finished dvc pull successfully")
    except Exception as e:
        print(f"Failed to do dvc pull: {e}")
