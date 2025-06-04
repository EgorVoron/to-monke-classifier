import logging
import os

import hydra
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.onnx
import torch.optim as optim
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
from omegaconf import DictConfig
from torcheval.metrics.functional import multiclass_f1_score

from to_monke_classifier.data.dataset import get_dataloaders
from to_monke_classifier.data.download import download_dvc_data
from to_monke_classifier.models.classifier import MonkeyClassifier
from to_monke_classifier.plotting import (
    save_accuracy_plot,
    save_f1_plot,
    save_loss_plot,
)
from to_monke_classifier.utils import get_git_commit_hash


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def run_training(cfg: DictConfig):
    download_dvc_data()

    train_loader, val_loader = get_dataloaders(
        train_dir=cfg.data.train_dir,
        val_dir=cfg.data.val_dir,
        batch_size=cfg.data.batch_size,
        img_size=cfg.data.image_size,
        num_workers=cfg.data.num_workers,
    )

    device = torch.device(cfg.training.device)
    model = MonkeyClassifier(
        num_classes=cfg.model.num_classes, fc_params=cfg.model.fc_params
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    )

    commit_id = get_git_commit_hash()

    mlflow.set_tracking_uri(cfg.training.mlflow.url)
    mlflow.set_experiment(cfg.training.mlflow.experiment_name)
    os.makedirs(cfg.training.plots.dir, exist_ok=True)

    train_losses, val_accs, f1_scores = [], [], []

    with mlflow.start_run(run_name="run_" + commit_id):
        mlflow.log_params(
            {
                "epochs": cfg.training.epochs,
                "batch_size": cfg.data.batch_size,
                "lr": cfg.training.lr,
                "weight_decay": cfg.training.weight_decay,
                "num_classes": cfg.model.num_classes,
                "fc_out_features": cfg.model.fc_params.linear_layer_out_features,
                "fc_dropout": cfg.model.fc_params.dropout_rate,
                "commit_id": commit_id,
            }
        )

        for epoch in range(cfg.training.epochs):
            model.train()
            running_loss = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            train_loss = running_loss / len(train_loader)
            train_losses.append(train_loss)
            logging.info(f"Epoch {epoch + 1}: Train loss: {train_loss:.4f}")

            model.eval()
            correct, total = 0, 0
            all_preds, all_labels = [], []
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                    all_preds.append(preds)
                    all_labels.append(labels)
            val_acc = correct / total * 100
            val_accs.append(val_acc)
            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)
            f1 = multiclass_f1_score(
                all_preds,
                all_labels,
                num_classes=cfg.model.num_classes,
                average="macro",
            ).item()
            f1_scores.append(f1)
            logging.info(
                f"Epoch {epoch + 1}: Val accuracy: {val_acc:.2f}%, F1 macro: {f1:.4f}"
            )

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)
            mlflow.log_metric("val_f1_macro", f1, step=epoch)

        save_loss_plot(train_losses, cfg.training.plots.file_loss)
        mlflow.log_artifact(cfg.training.plots.file_loss)

        save_accuracy_plot(val_accs, cfg.training.plots.file_acc)
        mlflow.log_artifact(cfg.training.plots.file_acc)

        save_f1_plot(f1_scores, cfg.training.plots.file_f1)
        mlflow.log_artifact(cfg.training.plots.file_f1)

        metrics_path = cfg.training.plots.metrics_path
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w") as f:
            for epoch in range(cfg.training.epochs):
                f.write(
                    f"Epoch {epoch + 1}: loss={train_losses[epoch]}, "
                    f"val_acc={val_accs[epoch]}, "
                    f"val_f1_macro={f1_scores[epoch]}\n"
                )
        mlflow.log_artifact(metrics_path)

        torch.save(model.state_dict(), cfg.training.checkpoint_file)
        input_schema = Schema(
            [
                TensorSpec(
                    np.dtype(np.float32),
                    (-1, 3, cfg.data.image_size, cfg.data.image_size),
                )
            ]
        )
        output_schema = Schema(
            [TensorSpec(np.dtype(np.float32), (-1, cfg.model.num_classes))]
        )
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        mlflow.pytorch.log_model(
            model, artifact_path=cfg.training.mlflow.artifact_path, signature=signature
        )

    logging.info("===== Training finished =====")

    onnx_path = cfg.training.onnx_file
    model.eval()
    dummy_input = torch.randn(
        1, 3, cfg.data.image_size, cfg.data.image_size, device=device
    )
    torch.onnx.export(
        model,
        (dummy_input,),
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    logging.info(f"model was exported to {onnx_path}")
    mlflow.log_artifact(onnx_path)


if __name__ == "__main__":
    run_training()
