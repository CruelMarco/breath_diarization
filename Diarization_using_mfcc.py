#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 14:22:02 2022

@author: shaique
"""
import IPython
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from tqdm import tqdm
import os
from scipy.io.wavfile import write
import scipy.signal
from spectrum import aryule
from pylab import plot, axis, xlabel, ylabel, grid, log10
import scipy.signal
from nara_wpe.wpe import wpe
from nara_wpe.wpe import get_power
from nara_wpe.utils import stft, istft, get_stft_center_frequencies
from nara_wpe import project_root
import os
import librosa
import pandas as pd
from pandas import DataFrame as df
import glob, os
import shutil
import json
import sklearn
import shutil
import math
from scipy import stats
from operator import itemgetter
import pickle
import warnings
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import *

# dir = '/home/shaique/Desktop/Shaique/mfcc_d_dd'

# os.chdir(dir)

# files = os.listdir(dir)

dir = 'C:/Users/Spirelab/Desktop/Breath_gender/only_breath_audios'

os.chdir(dir)

#os.chdir(dir)

breath_files = os.listdir(dir)

file = breath_files[3]

wavData,fs = librosa.load(file, sr = None)

ste = librosa.feature.rms(wavData,hop_length=int(16000/fs)).T

thresh = 0.01*(np.percentile(ste,97.5) + 9*np.percentile(ste,2.5)) 
   # Trim 5% off and set threshold as 0.1x of the ste range
indices = (ste>thresh).astype(float)

plot(wavData)

plot(indices)


