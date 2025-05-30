from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloaders(train_dir, val_dir, batch_size=16, img_size=224, num_workers=2):
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )
    train_data = datasets.ImageFolder(root=train_dir, transform=transform)
    val_data = datasets.ImageFolder(root=val_dir, transform=transform)
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader
