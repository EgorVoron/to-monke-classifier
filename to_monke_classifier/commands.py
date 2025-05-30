import fire
from to_monke_classifier import train, infer

def train_command():
    train.run_training()

def infer_command():
    return

if __name__ == "__main__":
    fire.Fire({
        "train": train_command,
        "infer": infer_command
    })