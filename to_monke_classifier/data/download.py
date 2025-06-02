from pathlib import Path
import logging
import dvc.api


def download_dvc_data(data_path: str):
    target = Path(data_path)
    if target.exists() and any(target.iterdir()):
        logging.info(f"{data_path} is downloaded already")
        return
    logging.info(f"{data_path} not found, downloading...")
    dvc.api.get(repo=".", path=data_path, out=str(target))
    logging.info(f"{data_path} was downloaded")
