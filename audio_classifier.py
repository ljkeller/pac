import os

from typing import Callable, Optional
from nicegui import events, ui

class AudioClassifier(ui.element, component='audio_classifier.vue'):
    def __init__(self) -> None:
        super().__init__()
        self.record_file = os.getenv('RECORD_FILE', default='record.wav')

    def classifiy_audio(self) -> None:
        raise NotImplementedError
