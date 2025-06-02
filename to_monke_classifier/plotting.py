import matplotlib.pyplot as plt
import os

def save_loss_plot(losses, out_path, title="Train Loss"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure()
    plt.plot(losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(title)
    plt.savefig(out_path)
    plt.close()

def save_accuracy_plot(accs, out_path, title="Val Accuracy"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure()
    plt.plot(accs, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title(title)
    plt.savefig(out_path)
    plt.close()

def save_f1_plot(f1_scores, out_path, title="Val F1 Macro"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure()
    plt.plot(f1_scores, label="Val F1 Macro")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.title(title)
    plt.savefig(out_path)
    plt.close()