import matplotlib.pyplot as plt
import os

def save_loss_acc_plot(
    losses, accuracies, out_path, title="Train Loss & Val Accuracy"
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure()
    plt.plot(losses, label="Train Loss")
    plt.plot(accuracies, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.title(title)
    plt.savefig(out_path)
    plt.close()
    print(f"Plot saved: {out_path}")