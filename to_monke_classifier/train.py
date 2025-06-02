import hydra
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig

from to_monke_classifier.data.dataset import get_dataloaders
from to_monke_classifier.data.download import download_dvc_data
from to_monke_classifier.models.classifier import MonkeyClassifier


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def run_training(cfg: DictConfig):
    download_dvc_data(cfg.data.train_dir)
    download_dvc_data(cfg.data.val_dir)

    train_loader, val_loader = get_dataloaders(
        train_dir=cfg.data.train_dir,
        val_dir=cfg.data.val_dir,
        batch_size=cfg.data.batch_size,
        img_size=cfg.data.image_size,
        num_workers=cfg.data.num_workers,
    )

    device = torch.device(cfg.training.device)
    model = MonkeyClassifier(num_classes=cfg.model.num_classes, fc_params=cfg.model.fc_params).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    )
    print("Staring training loop")
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
        print(f"Epoch {epoch + 1}, Train Loss: {running_loss / len(train_loader)}")

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total * 100
        print(f"Epoch {epoch + 1}, Val Accuracy: {val_acc:.2f}%")

    torch.save(model.state_dict(), "monkey_classifier.pth")
    print("Training finished! Model saved to monkey_classifier.pth")

if __name__ == "__main__":
    run_training()
