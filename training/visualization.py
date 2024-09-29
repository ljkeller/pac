import os


import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

n_fft = 1024 # librosa recommends 2048 for musical analysis and 512 for speech analysis

filename = os.getenv('AUDIO_FILE', default='sample.wav')
y, sample_rate = librosa.load(filename)

trimmed_sample, _ = librosa.effects.trim(y)

S = librosa.feature.melspectrogram(y=trimmed_sample, sr=sample_rate, n_fft=n_fft)
S_DB = librosa.power_to_db(S, ref=np.max)

print(f'Shape of S: {S.shape}')

fig, axs = plt.subplots(2)
fig.suptitle('Spectrogram, Waveform')

librosa.display.specshow(S_DB, sr=sample_rate, x_axis='time', y_axis='mel', ax=axs[0])
axs[0].set_title('Mel-frequency spectrogram')

fig.colorbar(mappable=axs[0].collections[0], ax=axs[0], format='%+2.0f dB', orientation='horizontal')

#plt.colorbar(format='%+2.0f dB')

librosa.display.waveshow(trimmed_sample, sr=sample_rate, ax=axs[1])
axs[1].set_title('Waveform')
axs[1].set_xlim([0, len(trimmed_sample) / sample_rate])

plt.show()
