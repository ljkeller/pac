"""runner.py: Batch process ML training jobs"""

import logging
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numexpr as ne
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from urbandata import UrbanSoundDataSet, k_fold_urban_sound, train_one_epoch, validate
from visualize import plot_final_results, plot_fold_results

logger = logging.getLogger(__name__)


def _get_jobs(jobs_path=Path("./jobs")):
    """Get all job files from the jobs directory"""

    # jobs starting with _ are ignored, like the default job
    return [
        f for f in jobs_path.iterdir() if f.is_file() and not f.name.startswith("_")
    ]


def run_jobs(jobs_path=Path("./jobs"), level=logging.INFO):
    """Run all jobs in the jobs directory"""

    for job_path in _get_jobs(jobs_path):
        # TODO: Move log into tmpdir?
        file_handler = logging.FileHandler(f"./{job_path.name}.log")
        file_handler.setLevel(level)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        try:
            logger.info(f"Starting job {job_path.name}.")
            with TrainingJob(job_path) as job:
                job.train()
            logger.info(f"Finished job {job_path.name}.")
        finally:
            logger.removeHandler(file_handler)
            file_handler.close()


class RuntimeNN(torch.nn.Module):
    """NN that is injected at runtime with a custom architecture"""

    def __init__(self, name, sequential_arch):
        super().__init__()
        self.name = name
        self.model = torch.nn.Sequential(*sequential_arch)

    def forward(self, X):
        return self.model(X)


def _evaluate_expression(value):
    """Evaluate an expression if it contains an operator"""

    logger.debug(f"Evaluating expression: {value}")
    try:
        if isinstance(value, str) and any(op in value for op in ["+", "-", "*", "/"]):
            return ne.evaluate(value).item()
        else:
            return value
    except Exception as e:
        raise ValueError("Failed to evaluate expression {}. {}".format(value, e))


class TrainingJob:
    """Encapsule ml training jobs with conventional Torch training styles & architectures"""

    def __init__(self, job_path):
        self.path = job_path
        self.job = {}
        self.start_time = 0.0
        self.layers = []
        self.fold_accuracies = []
        self.temp_dir = tempfile.TemporaryDirectory()

        self.momentum = 0.9
        self.dry_run = False
        self.learning_rate = 0.01
        self.sample_rate = None
        self.is_shuffled = True
        self.batch_size = 32
        self.epochs = 10

    def train(self):
        """Training, assuming job is on urbansound dataset"""

        urban_metadata = self.data_path / "metadata/UrbanSound8K.csv"
        urban_audio = self.data_path / "audio"

        folds = k_fold_urban_sound(urban_metadata, self.dry_run)

        logger.info(f"-----{len(folds)}-Fold Cross Validation-----")
        start_time = time.time()

        for fold_idx, fold_bundle in enumerate(tqdm(folds, desc="Fold progress")):
            logger.debug(f"<Fold {fold_idx}>")

            model = self.get_new_model()
            optimizer = torch.optim.SGD(
                model.parameters(), lr=self.learning_rate, momentum=self.momentum
            )

            train_ds = UrbanSoundDataSet(
                urban_audio,
                fold_bundle["train"],
                sample_rate=self.sample_rate,
                mel_kwargs=self.job["mel_kwargs"],
            )
            validation_ds = UrbanSoundDataSet(
                urban_audio,
                fold_bundle["validation"],
                sample_rate=self.sample_rate,
                mel_kwargs=self.job["mel_kwargs"],
            )
            logger.debug(
                f"\tSize of train, val datasets: {(len(train_ds), len(validation_ds))}"
            )

            train_dl = DataLoader(
                train_ds, batch_size=self.batch_size, shuffle=self.is_shuffled
            )
            validation_dl = DataLoader(
                validation_ds, batch_size=self.batch_size, shuffle=self.is_shuffled
            )

            losses_for_fold, accs_for_fold = [], []
            vacc = 0
            for _ in tqdm(range(self.epochs), desc="Epochs"):
                avg_loss, acc = train_one_epoch(
                    train_dl, model, optimizer, self.loss_fn, self.device
                )
                avg_vloss, vacc = validate(
                    validation_dl, model, self.loss_fn, self.device
                )

                losses_for_fold.append((avg_loss, avg_vloss))
                accs_for_fold.append((acc, vacc))
            #         print(f'LOSS train {avg_loss} val {avg_vloss}')
            logger.debug(f"Fold accuracy: {vacc*100:.2f}%")

            plot_fold_results(
                fold_idx,
                losses_for_fold,
                accs_for_fold,
                archive_path=Path(self.temp_dir.name),
            )
            self.fold_accuracies.append(vacc)

            logger.debug(f"</Fold {fold_idx}>")

        end_time = time.time()
        training_duration = end_time - start_time

        plot_final_results(self.fold_accuracies, archive_path=Path(self.temp_dir.name))
        self.kfold_valication_acc = np.mean(self.fold_accuracies)
        logger.info(f"Training time: {training_duration:.2f} seconds")

    def get_new_model(self):
        logger.debug("Getting new model.")
        return RuntimeNN(self.model_name, self.layers).to(self.device)

    def __enter__(self):
        self.start_time = time.time()

        with open(self.path, "r") as f:
            self.job = yaml.load(f, yaml.FullLoader)

        self._validate_job()
        self.__inject()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()

        # TODO: save job data: logs, pictures, model, etc... in completed dir.
        logger.info(
            "Job duration: {:.2f} seconds".format(self.end_time - self.start_time)
        )

        if exc_type is not None:
            self._failure_processing(exc_type, exc_value, traceback)
        else:
            self._success_processing()

        self.temp_dir.cleanup()

        return False

    def _validate_job(self):
        """
        Perform minimal/basic job validation by checking YAML configuration for inconsistencies.
        """

        assert "model" in self.job
        assert "name" in self.job["model"]
        assert "architecture" in self.job["model"]

        assert "ml_parameters" in self.job
        assert "epochs" in self.job["ml_parameters"]
        assert "audio_parameters" in self.job
        assert "job_parameters" in self.job

    def __inject(self):
        """Injects dependencies. Assumes inputs are validated / healthy"""
        self.model_name = self.job["model"]["name"]

        for layer in self.job["model"]["architecture"]:
            cls = getattr(torch.nn, layer["layer_type"])
            layer_params = {
                key: _evaluate_expression(value)
                for key, value in layer.items()
                if key != "layer_type"
            }
            self.layers.append(cls(**layer_params))
        logger.debug(f"Injected layers: {self.layers}")

        for key, value in self.job["ml_parameters"].items():
            setattr(self, key, value)
        for key, value in self.job["audio_parameters"].items():
            setattr(self, key, value)
        for key, value in self.job["job_parameters"].items():
            setattr(self, key, value)

        # Overrides for non-numeric / non-str types
        if getattr(self, "loss_fn"):
            self.loss_fn = getattr(torch.nn, self.loss_fn)()
            logger.debug(f"Using loss function: {self.loss_fn}")
        if getattr(self, "device"):
            self.device = torch.device(self.device)
            logger.info(f"Using device: {self.device}")
        if getattr(self, "data_path"):
            self.data_path = Path(self.data_path).expanduser()
            logger.debug(f"Using data path: {self.data_path}")

        logger.info(f"Injected job parameters: {self.__dict__}")

    def _failure_processing(self, exc_type, exc_value, traceback):
        logger.critical(f"Job failed with exception: {exc_type}.")
        logger.critical(f"Exception message: {exc_value}.")
        logger.critical(f"Traceback: {traceback}")

        dtime = datetime.now().strftime("%Y-%m-%d_%H-%M")
        exc = str(exc_type) if exc_type else "fail"
        dirname = f"{dtime}_{self.model_name}_E{self.epochs}_Exc{exc}"

        # TODO:
        # mkdir dirname
        # move contents from tmpdir to dirname

    def _success_processing(self):
        logger.info("Job trained succesfully- performing post-processing.")

        dtime = datetime.now().strftime("%Y-%m-%d_%H-%M")
        dirname = f"{dtime}_{self.model_name}_E{self.epochs}_Acc{self.kfold_valication_acc*100:.2f}"

        # TODO:
        # mkdir dirname
        # move contents from tmpdir to dirname
