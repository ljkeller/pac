import librosa
import numpy as np
from matplotlib import pyplot as plt

from urbandata import BASELINE_MODEL_ACCURACY


def plot_fold_results(foldidx, losses_for_fold, acc_for_fold):
    '''Plot the results of the model training for a single fold'''
    plt.figure(figsize=(10, 6))
    train_losses, val_losses = zip(*losses_for_fold)
    train_acc, val_acc = zip(*acc_for_fold)

    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel(f'Losses for fold {foldidx}')
    plt.title('Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(train_acc, label='Training Acc')
    plt.plot(val_acc, label='Validation Acc')
    plt.xlabel('Epochs')
    plt.ylabel(f'Accuracy for fold {foldidx}')
    plt.ylim((0, 1))
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_final_results(fold_accuracies):
    '''Plot the final results of the model training across all folds'''
    avg_accuracy = np.mean(fold_accuracies)

    plt.figure(figsize=(10, 6))

    plt.subplot(1, 2, 1)
    plt.plot(fold_accuracies, label='Fold Accuracies')

    plt.subplot(1, 2, 2)
    plt.axhline(y=int(BASELINE_MODEL_ACCURACY), color='r')
    plt.text(
        0,
        BASELINE_MODEL_ACCURACY+0.01,
        f'Baseline: {BASELINE_MODEL_ACCURACY:.2f}', color='r', ha='center'
    )
    plt.bar(['Avg Accuracy'], [avg_accuracy])
    plt.ylim((0, 1))
    plt.ylabel('Accuracy')


def plot_spectrogram(spectrogram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(librosa.power_to_db(spectrogram), origin="lower", aspect="auto", interpolation="nearest")
