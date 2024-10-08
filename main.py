#!/usr/bin/env python3
from audio_recorder import AudioRecorder
from audio_classifier import AudioClassifier

from nicegui import ui

with ui.row().classes('w-full justify-center'):
    audio_recorder = AudioRecorder(on_audio_ready=lambda data: ui.notify(f'Recorded {len(data)} bytes'))

with ui.row().classes('w-full justify-center'):
    ui.button('Play', on_click=audio_recorder.play_recorded_audio) \
        .bind_enabled_from(audio_recorder, 'recording')
    ui.button('Download', on_click=lambda: ui.download(audio_recorder.recording, 'record.wav')) \
        .bind_enabled_from(audio_recorder, 'recording')
    ui.button('Classify', on_click=lambda: print("classifying")) \
        .bind_enabled_from(audio_recorder, 'recording')

ui.run()
