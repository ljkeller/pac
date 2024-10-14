import re

import librosa
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
from torch.utils.data import Dataset

# print(f'Torch version: {torch.__version__}')
# print(f'Torchaudio version: {torchaudio.__version__}')
# print(f'Torch cuda enabled: {torch.cuda.is_available()}')
TARGET_SAMPLE_RATE = 22050
TARGET_DURATION = 4
LEN_IDEAL_WF = TARGET_DURATION * TARGET_SAMPLE_RATE

BASELINE_MODEL_ACCURACY = 0.68

def examine_urban_sound_df(df):
    '''Gather and print statistics about the UrbanSound8K dataset'''
    df_copy = df.copy()
    total = len(df_copy)

    print(f"Total samples: {total}")
    print(f"{'Class':<16} | {'Frequency':<10} | {'Percentage':<10}")
    print("-" * 40)

    distribution = df_copy['class'].value_counts()
    for cls, freq in distribution.items():
        percentage = (freq / total) * 100
        print(f"{cls:<16} | {freq:<10} | {percentage:.2f}%")
    print("-" * 40 + "\n")

    df_copy["duration"] = df_copy["end"] - df_copy["start"]
    print(f"Duration statistics: \n{df_copy['duration'].describe()}")


def train_one_epoch(dl, model, optimizer, loss_fn, device):
    model.train()

    running_loss = 0.
    running_batch_loss = 0.
    total = 0
    correct = 0
    # avg_batch_loss = 0

    for batch_idx, batch in enumerate(dl):
        (Xs, ys) = batch['spectrogram'].to(device), batch['label'].to(device)

        optimizer.zero_grad()

        yhats = model(Xs)
        _, yhats_as_idx = torch.max(yhats, 1)

        loss = loss_fn(yhats, ys)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        running_batch_loss += loss.item()

        total += ys.size(0)
        correct += (ys == yhats_as_idx).sum().item()

#         print(f'Training ground truth {ys}')
#         print(f'Training predictions {yhats_as_idx}')

#         if batch_idx % batch_print_threshold == batch_print_threshold-1:
#             last_loss = running_loss / batch_print_threshold #batch loss
#             avg_batch_loss = running_batch_loss / batch_print_threshold
#             print(f'\tbatch {batch_idx+1} loss: {avg_batch_loss}')
#             running_batch_loss = 0

    accuracy = correct / total
    return running_loss / (batch_idx+1), accuracy

def validate(validation_dl, model, loss_fn, device):
    model.eval()

    running_vloss = 0.
    total = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(validation_dl):
            (vXs, vys) = batch['spectrogram'].to(device), batch['label'].to(device)

            vyhats = model(vXs)
            _, vyhats_as_idx = torch.max(vyhats, 1)
            vloss = loss_fn(vyhats, vys)
            running_vloss += vloss

            total += vys.size(0)
            correct += (vys == vyhats_as_idx).sum().item()
#             print(f'Validation ground truth {vys}')
#             print(f'Validation redictions {vyhats_as_idx}')
    accuracy = correct/total
    return running_vloss.to('cpu') / (batch_idx+1), accuracy


def plot_fold_results(foldidx, losses_for_fold, acc_for_fold):
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
    plt.ylim((0, BASELINE_MODEL_ACCURACY))
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_final_results(fold_accuracies):
    avg_accuracy = np.mean(fold_accuracies)
    
    plt.figure(figsize=(10,6))
    
    plt.subplot(1, 2, 1)
    plt.plot(fold_accuracies, label='Fold Accuracies')
    
    plt.subplot(1, 2, 2)
    plt.axhline(y=baseline_model_accuracy, color='r')
    plt.text(0, baseline_model_accuracy+0.01, f'Baseline: {baseline_model_accuracy:.2f}', color='r', ha='center')
    plt.bar(['Avg Accuracy'], [avg_accuracy])
    plt.ylabel('Accuracy')


def k_fold_urban_sound(metadata_path, dry_run=False):
    """
    Extract the 10 recommended folds of UrbanSound8K. Combine the validation training accuracy from each fold
    to get a more accurate estimate of the model's performance.

    Args:
        metadata_path (str): Path to the UrbanSound8K metadata file.
        dry_run (bool): If True, the function will only yield 5% of the data for testing purposes.

    Returns:
        a list of map folds in the form:
    [{train: [fold_1_training_data_paths], validation: [fold_1_validation_data_paths]}),
     {train: [fold_2_training_data_paths], validation: [fold_2_validation_data_paths]}),
     ...
     {train: [fold_10_training_data_paths], validation: [fold_10_validation_data_paths]}),
     ]

    """
    folds = []
    frame = pd.read_csv(metadata_path)
    frame.info()

    print("\nSummarizing folds:")
    print('-----------------------------------------------------------')
    for i in range(1, 11):
        train_mask = frame['fold'] != i
        validation_mask = frame['fold'] == i
        # TODO: Duration mask?

        print(f'Training set size for fold {i} : {len(frame[train_mask])}')
        train = frame[train_mask]
        print("Training set info: \n")
        examine_urban_sound_df(train)

        print(f'Validation set size for fold {i} : {len(frame[validation_mask])}')
        validation = frame[validation_mask]
        print('Validation set info: \n')
        examine_urban_sound_df(validation)

        train_paths = train.apply(lambda r: f"fold{r['fold']}/{r['slice_file_name']}", axis=1)
        validation_paths = validation.apply(
            lambda r: f"fold{r['fold']}/{r['slice_file_name']}", axis=1
        )

        folds.append(
            {'train': train_paths.tolist()[:len(train_paths)//20 if dry_run else None],
             'validation': validation_paths.tolist()[:len(validation_paths)//20 if dry_run else None]}
        )
        print('-----------------------------------------------------------')
    print("\n\n")

    return folds


class Rescale(object):
    """Rescale an audio signal to given length via cropping or padding

    Args:
        output_size (int): Desired output size.
    """

    def __init__(self, output_size):
        self.output_size = output_size
        assert isinstance(output_size, int)

    def __call__(self, waveform):
        waveform = torch.squeeze(waveform)
        assert waveform.ndim <= 2

        # TODO: Pull stereo / mono into new class?

        ret_wf = None
        if waveform.ndim == 1:
            waveform = torch.stack((waveform, waveform))

        if len(waveform[0]) < self.output_size:
            ret_wf = F.pad(waveform, (0, self.output_size-len(waveform[0]), 0, 0))
        elif len(waveform[0]) > self.output_size:
            expanded_l = waveform[0][:self.output_size]
            expanded_r = waveform[1][:self.output_size]
            ret_wf = torch.stack((expanded_l, expanded_r))
        else:
            ret_wf = waveform

        return ret_wf


class UrbanSoundDataSet(Dataset):
    def __init__(self, urban_audio_path, relativepaths, transform=None, sample_rate=None, mel_kwargs=None):
        self.sounds = list({urban_audio_path/path for path in relativepaths})
        self.resampled_sample_rate = sample_rate
        self.transform = transform
        self.mel_kwargs = mel_kwargs if mel_kwargs is not None else {}

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sound_fp = self.sounds[idx]

        match = re.search(r'\d+-(\d)-\d+-\d+\.wav$', str(sound_fp))
        label = torch.tensor(int(match.group(1)) if match else -1)

        # normalize here is converting native sample type to f32
        waveform, native_sample_rate = torchaudio.load(sound_fp, normalize=True)

        transforms = self._get_transforms(native_sample_rate)
        for t in transforms:
            waveform = t(waveform)
        mel_spectrogram = waveform

        sample = {"spectrogram": mel_spectrogram, "label": label}
        return sample

    def _get_transforms(self, native_sr):
        return [
            T.Resample(native_sr, self.resampled_sample_rate),
            Rescale(LEN_IDEAL_WF),
            T.MelSpectrogram(self.resampled_sample_rate, **self.mel_kwargs)
        ]

    def getXShape(self):
        '''Return the common shape of sample data (after preprocessing)'''

        # This method is robust to changes in torch defaults, but its annoying we have to load a sample
        sound_fp = self.sounds[0]
        wf, native_sr = torchaudio.load(sound_fp, normalize=True)
        tforms = self._get_transforms(native_sr)

        final_wf = wf
        for t in tforms:
            final_wf = t(final_wf)

        return final_wf.shape

    def __len__(self):
        return len(self.sounds)


def plot_spectrogram(spectrogram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(librosa.power_to_db(spectrogram), origin="lower", aspect="auto", interpolation="nearest")
