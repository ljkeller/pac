"""Module for visualizing the results of the model training"""

from pathlib import Path
from typing import Optional

import librosa
import numpy as np
from matplotlib import pyplot as plt
from urbandata import BASELINE_MODEL_ACCURACY


def plot_fold_results(
    foldidx, losses_for_fold, acc_for_fold, archive_path: Optional[Path] = None
):
    """Plot the results of the model training for a single fold

    Args:
        foldidx (int): The index of the fold between [0...n-1]
        losses_for_fold (list): A list of tuples containing the training and validation losses
        acc_for_fold (list): A list of tuples containing the training and validation accuracies
        archive_path (Path): The option path to the archive directory for saving the plot

    Note: if archive_path is provided, the plot will be saved to the archive_path
    """
    fold = foldidx + 1

    plt.figure(figsize=(10, 6))
    train_losses, val_losses = zip(*losses_for_fold)
    train_acc, val_acc = zip(*acc_for_fold)

    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel(f"Losses for fold {fold}")
    plt.title("Loss")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(train_acc, label="Training Acc")
    plt.plot(val_acc, label="Validation Acc")
    plt.xlabel("Epochs")
    plt.ylabel(f"Accuracy for fold {fold}")
    plt.ylim((0, 1))
    plt.legend()

    plt.tight_layout()
    plt.show()

    if archive_path and archive_path.exists():
        plt.savefig(archive_path / f"fold_{fold}_results.png")


def plot_final_results(fold_accuracies, archive_path: Optional[Path] = None):
    """Plot the final results of the model training across all folds

    Args:
        fold_accuracies (list): A list of accuracies for each fold
        archive_path (Path): The option path to the archive directory for saving the plot

    Note: if archive_path is provided, the plot will be saved to the archive_path
    """

    avg_accuracy = np.mean(fold_accuracies)

    plt.figure(figsize=(10, 6))

    plt.subplot(1, 2, 1)
    plt.plot(fold_accuracies, label="Fold Accuracies")

    plt.subplot(1, 2, 2)
    plt.axhline(y=int(BASELINE_MODEL_ACCURACY), color="r")
    plt.text(
        0,
        BASELINE_MODEL_ACCURACY + 0.01,
        f"Baseline: {BASELINE_MODEL_ACCURACY:.2f}",
        color="r",
        ha="center",
    )
    plt.bar(["Avg Accuracy"], [avg_accuracy])
    plt.ylim((0, 1))
    plt.ylabel("Accuracy")

    plt.tight_layout()
    if archive_path and archive_path.exists():
        plt.savefig(archive_path / "final_results.png")


def plot_spectrogram(spectrogram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(
        librosa.power_to_db(spectrogram),
        origin="lower",
        aspect="auto",
        interpolation="nearest",
    )
