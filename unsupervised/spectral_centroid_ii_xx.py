# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 20:00:42 2023

@author: Spirelab
"""

import librosa
import numpy as np
import os
import pandas as pd

audio_dir = 'C:/Users/Spirelab/Desktop/Breath_Diarization/only_breath_audios'

audios = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]

os.chdir(audio_dir)

annotes = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]

file = audios[0]

file,sr = librosa.load(file, sr = 16000)

hop_length = 320

n_fft = 2048

stft = librosa.stft(file, n_fft= 2048 , hop_length = 320)

#Calc magnitude spectrum

spec = np.abs(stft)

##calc spectral centroid frame-wise

spectral_centroids = librosa.feature.spectral_centroid(S=spec)

window_size = int(np.ceil(sr / hop_length)) * 2 + 1

spectral_centroids_smooth = np.convolve(spectral_centroids[0], np.ones(window_size), 'same') / window_size

threshold = np.median(spectral_centroids_smooth)

inhale = np.where(spectral_centroids_smooth > threshold)[0]

exhale = np.where(spectral_centroids_smooth < threshold)[0]

frame_times = librosa.frames_to_time(np.arange(len(spectral_centroids[0])), sr=sr, hop_length=hop_length)

inhale_times = frame_times[inhale]

exhale_times = frame_times[exhale]
# Print the results
print(f'Inhale Time: {inhale}')

print(f'Exhale Time: {exhale}')