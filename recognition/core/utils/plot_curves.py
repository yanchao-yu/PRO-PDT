import numpy as np
import matplotlib.pyplot as plt
import os

def plot_curves(train_losses, train_accs, val_accs, model_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # loss on left y-axis
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='steelblue')
    ax1.plot(epochs, train_losses, color='steelblue', linewidth=1.5, label='Train Loss')
    ax1.tick_params(axis='y', labelcolor='steelblue')

    # accuracy on right y-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy (%)', color='coral')
    ax2.plot(epochs, train_accs, color='coral', linewidth=1.5, label='Train Accuracy')
    if val_accs:
        ax2.plot(epochs, val_accs, color='green', linewidth=1.5,
                 linestyle='--', label='Val Accuracy')
    ax2.tick_params(axis='y', labelcolor='coral')

    # combine legends from both axes into one
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    plt.title(f'{model_name} — Loss and Accuracy')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(save_dir, f"{model_name}_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Curves saved to {path}")


def plot_comparison(cnn_losses, cnn_accs, vit_losses, vit_accs, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # CNN plot
    ax1_loss = ax1
    ax1_acc = ax1.twinx()

    ax1_loss.set_xlabel('Epoch')
    ax1_loss.set_ylabel('Loss', color='steelblue')
    ax1_loss.plot(range(1, len(cnn_losses)+1), cnn_losses,
                  color='steelblue', linewidth=1.5, label='Loss')
    ax1_loss.tick_params(axis='y', labelcolor='steelblue')

    ax1_acc.set_ylabel('Accuracy (%)', color='coral')
    ax1_acc.plot(range(1, len(cnn_accs)+1), cnn_accs,
                 color='coral', linewidth=1.5, label='Accuracy')
    ax1_acc.tick_params(axis='y', labelcolor='coral')

    lines1, labels1 = ax1_loss.get_legend_handles_labels()
    lines2, labels2 = ax1_acc.get_legend_handles_labels()
    ax1_loss.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    ax1.set_title('CNN (Inception-ResNet-V1)')
    ax1.grid(True, alpha=0.3)

    # ViT plot
    ax2_loss = ax2
    ax2_acc = ax2.twinx()

    ax2_loss.set_xlabel('Epoch')
    ax2_loss.set_ylabel('Loss', color='steelblue')
    ax2_loss.plot(range(1, len(vit_losses)+1), vit_losses,
                  color='steelblue', linewidth=1.5, label='Loss')
    ax2_loss.tick_params(axis='y', labelcolor='steelblue')

    ax2_acc.set_ylabel('Accuracy (%)', color='coral')
    ax2_acc.plot(range(1, len(vit_accs)+1), vit_accs,
                 color='coral', linewidth=1.5, label='Accuracy')
    ax2_acc.tick_params(axis='y', labelcolor='coral')

    lines3, labels3 = ax2_loss.get_legend_handles_labels()
    lines4, labels4 = ax2_acc.get_legend_handles_labels()
    ax2_loss.legend(lines3 + lines4, labels3 + labels4, loc='center right')
    ax2.set_title('ViT (Face Transformer)')
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Training Curves: CNN vs ViT', fontsize=14, fontweight='bold')
    plt.tight_layout()

    path = os.path.join(save_dir, "CNN_vs_ViT_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Comparison plot saved to {path}")

