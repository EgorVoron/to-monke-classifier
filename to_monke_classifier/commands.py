import logging

import fire
from omegaconf import OmegaConf

from to_monke_classifier.infer import run_inference
from to_monke_classifier.train import run_training


def train_command():
    run_training()


def infer_command(image_path, config_path="configs/infer.yaml"):
    cfg = OmegaConf.load(config_path)
    logging.info(
        run_inference(
            image_path, cfg.server.url, cfg.data.image_size, cfg.data.labels_info_path
        )
    )


if __name__ == "__main__":
    fire.Fire({"train": train_command, "infer": infer_command})
