
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


def save_training_plot(history, output_dir):

    epochs = history["epoch"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Training and validation loss
    axes[0].plot(
        epochs,
        history["train_loss"],
        label="Training loss",
    )

    axes[0].plot(
        epochs,
        history["val_loss"],
        label="Validation loss",
    )

    axes[0].set_title("Training and Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Plot 2: Segmentation performance
    axes[1].plot(
        epochs,
        history["val_iou"],
        label="Validation IoU",
    )

    axes[1].plot(
        epochs,
        history["val_f1"],
        label="Validation F1",
    )

    axes[1].set_title("Segmentation Performance")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()

    fig.savefig(
        output_dir / "training_curves.png",
        dpi=150,
    )

    plt.close(fig)
