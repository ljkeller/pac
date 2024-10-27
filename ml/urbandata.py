import io
import logging
import re

import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

logging.debug(f"Torch version: {torch.__version__}")
logging.debug(f"Torchaudio version: {torchaudio.__version__}")
logging.debug(f"Torch cuda enabled: {torch.cuda.is_available()}")
TARGET_SAMPLE_RATE = 22050
TARGET_DURATION = 4

BASELINE_MODEL_ACCURACY = 0.68


def examine_urban_sound_df(df):
    """Gather and print statistics about the UrbanSound8K dataset"""
    df_copy = df.copy()
    total = len(df_copy)

    print(f"Total samples: {total}")
    print(f"{'Class':<16} | {'Frequency':<10} | {'Percentage':<10}")
    print("-" * 40)

    distribution = df_copy["class"].value_counts()
    for cls, freq in distribution.items():
        percentage = (freq / total) * 100
        print(f"{cls:<16} | {freq:<10} | {percentage:.2f}%")
    print("-" * 40 + "\n")

    df_copy["duration"] = df_copy["end"] - df_copy["start"]
    print(f"Duration statistics: \n{df_copy['duration'].describe()}")


def train_one_epoch(dl, model, optimizer, loss_fn, device):
    logger.debug("Training one epoch")
    model.train()

    running_loss = 0.0
    running_batch_loss = 0.0
    total = 0
    correct = 0
    # avg_batch_loss = 0

    for batch_idx, batch in enumerate(dl):
        (Xs, ys) = batch["spectrogram"].to(device), batch["label"].to(device)
        logger.debug(f"Batch {batch_idx} | Xs shape: {Xs.shape} | ys shape: {ys.shape}")

        optimizer.zero_grad()

        yhats = model(Xs)
        _, yhats_as_idx = torch.max(yhats, 1)
        logger.debug(f"Training ground truth: {ys}")
        logger.debug(f"Training predictions: {yhats_as_idx}")

        loss = loss_fn(yhats, ys)
        loss.backward()
        logger.debug(f"Training epoch loss: {loss.item()}")

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
    logger.debug(f"Epoch accuracy: {accuracy*100:.2f}%")
    return running_loss / (batch_idx + 1), accuracy


def validate(validation_dl, model, loss_fn, device):
    model.eval()

    running_vloss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(validation_dl):
            (vXs, vys) = (batch["spectrogram"].to(device), batch["label"].to(device))

            vyhats = model(vXs)
            _, vyhats_as_idx = torch.max(vyhats, 1)
            logger.debug(f"Validation ground truth: {vys}")
            logger.debug(f"Validation predictions: {vyhats_as_idx}")
            vloss = loss_fn(vyhats, vys)
            running_vloss += vloss

            total += vys.size(0)
            correct += (vys == vyhats_as_idx).sum().item()
            logger.debug(f"Correct: {correct} | Total: {total}")
    accuracy = correct / total
    logger.debug(f"Validation accuracy: {accuracy*100:.2f}%")

    return running_vloss.to("cpu") / (batch_idx + 1), accuracy


def k_fold_urban_sound(metadata_path, dry_run=False):
    """
    Extract the 10 recommended folds of UrbanSound8K. Combine the validation
    training accuracy from each fold to get a more accurate estimate of the
    model's performance.

    Args:
        metadata_path (str): Path to the UrbanSound8K metadata file.
        dry_run (bool): If True, the function will only yield 5% of the data
            for testing purposes.

    Returns:
        a list of map folds in the form:
    [{train: [fold_1_training_data_paths], validation: [fold_1_validation_data_paths]}),
     {train: [fold_2_training_data_paths], validation: [fold_2_validation_data_paths]}),
     ...
     {train: [fold_10_training_data_paths], validation: [fold_10_validation_data_paths]}),
     ]

    """
    folds = []
    logger.debug(f"Reading UrbanSound8K metadata from: {metadata_path}")
    frame = pd.read_csv(metadata_path)
    logger.info("UrbanSound8K metadata:")

    # redirect workaround for pd.DataFrame.info()
    buffer = io.StringIO()
    frame.info(buf=buffer)
    info_str = buffer.getvalue()
    logger.info(info_str)

    logger.info("\nSummarizing folds:")
    logger.info("-----------------------------------------------------------")
    for i in range(1, 11):
        train_mask = frame["fold"] != i
        validation_mask = frame["fold"] == i
        # TODO: Duration mask?

        logger.info(f"Training set size for fold {i} : {len(frame[train_mask])}")
        train = frame[train_mask]
        logger.info("Training set info: \n")
        examine_urban_sound_df(train)

        logger.info(f"Validation set size for fold {i} : {len(frame[validation_mask])}")
        validation = frame[validation_mask]
        logger.info("Validation set info: \n")
        examine_urban_sound_df(validation)

        train_paths = train.apply(
            lambda r: f"fold{r['fold']}/{r['slice_file_name']}", axis=1
        )
        validation_paths = validation.apply(
            lambda r: f"fold{r['fold']}/{r['slice_file_name']}", axis=1
        )

        folds.append(
            {
                "train": train_paths.tolist()[
                    : len(train_paths) // 20 if dry_run else None
                ],
                "validation": validation_paths.tolist()[
                    : len(validation_paths) // 20 if dry_run else None
                ],
            }
        )
        logger.info("-----------------------------------------------------------")
    logger.info("\n\n")

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
            ret_wf = F.pad(waveform, (0, self.output_size - len(waveform[0]), 0, 0))
        elif len(waveform[0]) > self.output_size:
            expanded_l = waveform[0][: self.output_size]
            expanded_r = waveform[1][: self.output_size]
            ret_wf = torch.stack((expanded_l, expanded_r))
        else:
            ret_wf = waveform

        return ret_wf


class UrbanSoundDataSet(Dataset):
    def __init__(
        self,
        urban_audio_path,
        relativepaths,
        transform=None,
        sample_rate=None,
        mel_kwargs=None,
        target_duration=4,
    ):
        self.sounds = list({urban_audio_path / path for path in relativepaths})
        self.resampled_sample_rate = int(sample_rate)
        self.target_duration = target_duration
        self.transform = transform
        self.mel_kwargs = mel_kwargs if mel_kwargs is not None else {}
        logger.debug(f"Mel kwargs: {self.mel_kwargs}")
        logger.debug(f"Sample rate: {self.resampled_sample_rate}")
        logger.debug(f"Target duration: {self.target_duration}")

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sound_fp = self.sounds[idx]

        match = re.search(r"\d+-(\d)-\d+-\d+\.wav$", str(sound_fp))
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
        len_ideal_wf = self.target_duration * self.resampled_sample_rate
        return [
            T.Resample(native_sr, self.resampled_sample_rate),
            Rescale(len_ideal_wf),
            T.MelSpectrogram(self.resampled_sample_rate, **self.mel_kwargs),
        ]

    def getXShape(self):
        """Return the common shape of sample data (after preprocessing)"""

        # This method is robust to changes in torch defaults,
        # but its annoying we have to load a sample
        sound_fp = self.sounds[0]
        wf, native_sr = torchaudio.load(sound_fp, normalize=True)
        tforms = self._get_transforms(native_sr)

        final_wf = wf
        for t in tforms:
            final_wf = t(final_wf)

        return final_wf.shape

    def __len__(self):
        return len(self.sounds)
