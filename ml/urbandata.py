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


def k_fold_urban_sound(metadata_path):
    """
    Extract the 10 recommended folds of UrbanSound8K

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

        folds.append({'train': train_paths.tolist(), 'validation': validation_paths.tolist()})
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
        label = match.group(1) if match else -1

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
