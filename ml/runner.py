'''runner.py: Batch process ML training jobs'''

from pathlib import Path
import time
from datetime import datetime

import numexpr as ne
import torch
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import yaml

from urbandata import k_fold_urban_sound, UrbanSoundDataSet, train_one_epoch, validate
from visualize import plot_fold_results, plot_final_results


def get_jobs(jobs_path=Path('./jobs')):
    '''Get all job files from the jobs directory'''

    # jobs starting with _ are ignored, like the default job
    return [f for f in jobs_path.iterdir() if f.is_file() and not f.name.startswith('_')]


class RuntimeNN(torch.nn.Module):
    '''NN that is injected at runtime with a custom architecture'''

    def __init__(self, name, sequential_arch):
        super().__init__()
        self.name = name
        self.model = torch.nn.Sequential(*sequential_arch)

    def forward(self, X):
        return self.model(X)

def evaluate_expression(value):
    '''Evaluate an expression if it contains an operator'''

    try: 
        if isinstance(value, str) and any(op in value for op in ['+', '-', '*', '/']):
            return ne.evaluate(value).item()
        else:
            return value
    except Exception as e:
        raise ValueError('Failed to evaluate expression {}. {}'.format(value, e))


class TrainingJob():
    '''Encapsule ml training jobs with conventional Torch training styles & architectures'''

    def __init__(self, job_path):
        self.path = job_path
        self.job = None
        self.start_time = 0.
        self.layers = []
        self.fold_accuracies = []

    def train(self):
        '''Training, assuming job is on urbansound dataset'''

        urban_metadata = self.data_path / 'metadata/UrbanSound8K.csv'
        urban_audio = self.data_path / 'audio'

        folds = k_fold_urban_sound(urban_metadata, self.dry_run)

        print(f'-----{len(folds)}-Fold Cross Validation-----')
        start_time = time.time()

        for fold_idx, fold_bundle in enumerate(tqdm(folds, desc='Fold progress')):
            print(f"Fold {fold_idx}:", end='')

            model = self.get_new_model()
            optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate, momentum=self.momentum)

            train_ds = UrbanSoundDataSet(urban_audio,
                                         fold_bundle['train'],
                                         sample_rate=self.sample_rate,
                                         mel_kwargs=self.job['mel_kwargs'])
            validation_ds = UrbanSoundDataSet(urban_audio,
                                              fold_bundle['validation'],
                                              sample_rate=self.sample_rate,
                                              mel_kwargs=self.job['mel_kwargs'])
            print(f"\tSize of train, val datasets: {(len(train_ds), len(validation_ds))}")

            train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=self.is_shuffled)
            validation_dl = DataLoader(validation_ds, batch_size=self.batch_size, shuffle=self.is_shuffled)

            losses_for_fold, accs_for_fold = [], []
            for epoch in tqdm(range(self.epochs), desc='Epochs'):
                avg_loss, acc = train_one_epoch(train_dl, model, optimizer, self.loss_fn, self.device)
                avg_vloss, vacc = validate(validation_dl, model, self.loss_fn, self.device)

                losses_for_fold.append((avg_loss, avg_vloss))
                accs_for_fold.append((acc, vacc))
        #         print(f'LOSS train {avg_loss} val {avg_vloss}')
            print(f'Fold accuracy: {vacc*100:.2f}%')

            plot_fold_results(fold_idx, losses_for_fold, accs_for_fold)
            fold_accuracies.append(vacc)

        end_time = time.time()
        training_duration = end_time - start_time

        # TODO: move this to success processing?
        plot_final_results(self.fold_accuracies)
        self.kfold_valication_acc = np.mean(fold_accuracies)
        print(f"Training time: {training_duration:.2f} seconds")

    def get_new_model(self):
        return CustomNN(self.model_name, self.layers).to(self.device)

    def __enter__(self):
        self.start_time = time.time()

        with open(self.path, 'r') as f:
            self.job = yaml.load(f, yaml.FullLoader)

        self._validate_job()
        self.__inject()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()

        # TODO: save job data: logs, pictures, model, etc... in completed dir.
        print('Job duration: {:.2f} seconds'.format(self.end_time-self.start_time))

        if exc_type is not None:
            self._failure_processing(exc_type)
        else:
            self._success_processing()

        return False

    def _validate_job(self):
        '''
        Perform minimal/basic job validation by checking YAML configuration for inconsistencies.
        '''

        assert 'model' in self.job
        assert 'name' in self.job['model']
        assert 'architecture' in self.job['model']

        assert 'ml_parameters' in self.job
        assert 'epochs' in self.job['ml_parameters']
        assert 'audio_parameters' in self.job
        assert 'job_parameters' in self.job

    def __inject(self):
        '''Injects dependencies. Assumes inputs are validated / healthy'''
        self.model_name = self.job['model']['name']
        
        for layer in self.job['model']['architecture']:
            cls = getattr(torch.nn, layer['layer_type'])
            layer_params = {key: evaluate_expression(value) for key, value in layer.items() if key != 'layer_type'}
            self.layers.append(cls(**layer_params))

        for key, value in self.job['ml_parameters'].items():
            setattr(self, key, value)
        for key, value in self.job['audio_parameters'].items():
            setattr(self, key, value)
        for key, value in self.job['job_parameters'].items():
            setattr(self, key, value)

        # Overrides for non-numeric / non-str types
        if getattr(self, 'loss_fn'):
            self.loss_fn = getattr(torch.nn, self.loss_fn)()
        if getattr(self, 'device'):
            self.device = torch.device(self.device)
        if getattr(self, 'data_path'):
            self.data_path = Path(self.data_path).expanduser()

    def _failure_processing(self, exc_type):
        dtime = datetime.now().strftime("%Y-%m-%d_%H-%M")
        exc = str(exc_type) if exc_type else "fail"
        dirname = f"{dtime}_{self.model_name}_E{self.epochs}_Exc{exc}"

        # mkdir dirname
        # move contents from './tmp/<hash>' to dirname

        raise notImplementedError

    def _success_processing(self):
        dtime = datetime.now().strftime("%Y-%m-%d_%H-%M")
        dirname = f"{dtime}_{self.model_name}_E{self.epochs}_Acc{kfold_valication_acc*100:.2f}"

        # mkdir dirname
        # move contents from './tmp/<hash>' to dirname

        raise notImplementedError
