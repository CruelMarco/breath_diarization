# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 17:41:35 2022

@author: 91761
"""

import IPython
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from tqdm import tqdm
import os
import scipy.signal
import scipy.signal
import os
import librosa
import pandas as pd
from pandas import DataFrame as df
import glob, os
import json
import sklearn
import shutil
import math
from scipy import stats
from operator import itemgetter
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from numpy import loadtxt
import xgboost
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import librosa
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Dropout, Input
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks
from tensorflow.keras.layers import BatchNormalization
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD, RMSprop
from keras import optimizers
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, Bidirectional
from keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint,TensorBoard,ProgbarLogger
from keras.utils import np_utils
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import itertools
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten,Dropout,MaxPooling1D,Activation
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import seaborn as sns
from sklearn.metrics import confusion_matrix

#model_save_dir = 'C:/IISC_MODELS'


############# Data Preprocessing #############

def encoder(mfcc):
   
    test_set = np.array(mfcc)
   
    arr_y = []
   
    for x in test_set:
       
        for y in x:
           
            if y =='ii':
                arr_y.append([0])
            else:
                arr_y.append([1])
    arr_y=np.array(arr_y)
   
    return(arr_y)

def reshaper(mfcc):
   
    #mfcc = mfcc[np.random.default_rng(seed=42).permutation(mfcc.columns.values)]
   
    #mfcc = mfcc.loc[:, mfcc.columns.drop(['F0_mean', 'F0_median', 'F0_mode', 'F0_std'])]
   
   
    set_train_rs = np.array(mfcc)
   
    set_train_rs = set_train_rs.reshape(set_train_rs.shape[0], set_train_rs.shape[1],1)
   
    set_train_rs = np.array(set_train_rs)
   
    return(set_train_rs)

def plotter(mfcc):
    plt.plot(mfcc.history['accuracy'])
    plt.plot(mfcc.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    plt.plot(mfcc.history['loss'])
    plt.plot(mfcc.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
   
def plot_confusion5(cf):
   
    ax = sns.heatmap(cf, annot=True, cmap='Blues',fmt = ".1f")
    ax.set_title('CM FOR FOLD 5 with labels\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');
    ax.xaxis.set_ticklabels(['II','XX'])
    ax.yaxis.set_ticklabels(['II','XX'])
    plt.show()
   
def plot_confusion4(cf):
   
    ax = sns.heatmap(cf, annot=True, cmap='Blues',fmt = ".1f")
    ax.set_title('CM FOR FOLD 4 with labels\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');
    ax.xaxis.set_ticklabels(['II','XX'])
    ax.yaxis.set_ticklabels(['II','XX'])
    plt.show()

def plot_confusion3(cf):
   
    ax = sns.heatmap(cf, annot=True, cmap='Blues',fmt = ".1f")
    ax.set_title('CM FOR FOLD 3 with labels\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');
    ax.xaxis.set_ticklabels(['II','XX'])
    ax.yaxis.set_ticklabels(['II','XX'])
    plt.show()
def plot_confusion2(cf):
   
    ax = sns.heatmap(cf, annot=True, cmap='Blues',fmt = ".1f")
    ax.set_title('CM FOR FOLD 2 with labels\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');
    ax.xaxis.set_ticklabels(['II','XX'])
    ax.yaxis.set_ticklabels(['II','XX'])
    plt.show()
   
def plot_confusion1(cf):
   
    ax = sns.heatmap(cf, annot=True, cmap='Blues',fmt = ".1f")
    ax.set_title('CM FOR FOLD 1 with labels\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');
    ax.xaxis.set_ticklabels(['II','XX'])
    ax.yaxis.set_ticklabels(['II','XX'])
    plt.show()
   
   
dir = 'C:/mfcc_iixx_d_dd/50_frames/dl_sets/cnn_sets/fold1'

os.chdir(dir)

files2 = os.listdir(dir)

train_set_dir2 = os.path.join(dir,files2[1])

train_set2 = os.listdir(train_set_dir2)

val_set_dir2 = os.path.join(dir,files2[2])

val_set2 = os.listdir(val_set_dir2)

test_set_dir2 = os.path.join(dir,files2[0])

test_set2 = os.listdir(test_set_dir2)

###### Train Set Creation #####

train_set_mfcc2 = []

for i in train_set2:
   
    train_sub_mfcc_dir2 = os.path.join(train_set_dir2, i)
   
    train_sub_mfcc2 = pd.read_csv(train_sub_mfcc_dir2, sep = ',')
   
    train_set_mfcc2.append(train_sub_mfcc2)

train_set_mfcc2 = pd.concat(train_set_mfcc2)

#train_set_mfcc2 = train_set_mfcc2.sample(frac = 1)

train_set_y_2 = train_set_mfcc2.loc[:, train_set_mfcc2.columns.drop(['Name','Unnamed: 0','phon_idx','F0_d','F1_d','F2_d','F3_d','F4_d','F5_d','F6_d','F7_d','F8_d','F9_d','F10_d','F11_d','F12_d','F0_dd','F1_dd','F2_dd','F3_dd','F4_dd','F5_dd','F6_dd','F7_dd','F8_dd','F9_dd','F10_dd','F11_dd','F12_dd','F0','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12'])]

train_set_y_2_1 = encoder(train_set_y_2)

train_set_Inhale_count2 = np.count_nonzero(train_set_y_2_1 == 0)

train_set_Exhale_count2 = np.count_nonzero(train_set_y_2_1 == 1)

train_set_x_2 = train_set_mfcc2.loc[:, train_set_mfcc2.columns.drop(['Name','Unnamed: 0','phon_idx','phon'])]

train_set_x_2_1 = reshaper(train_set_x_2)  
###### Val Set Creation #####

val_set_mfcc2 = []

for j in val_set2:
   
    val_sub_mfcc_dir2 = os.path.join(val_set_dir2, j)
   
    val_sub_mfcc2 = pd.read_csv(val_sub_mfcc_dir2, sep = ',')
   
    val_set_mfcc2.append(val_sub_mfcc2)

val_set_mfcc2 = pd.concat(val_set_mfcc2)

#val_set_mfcc2 = val_set_mfcc2.sample(frac = 1)

val_set_y_2 = val_set_mfcc2.loc[:, val_set_mfcc2.columns.drop(['Name','Unnamed: 0','phon_idx','F0_d','F1_d','F2_d','F3_d','F4_d','F5_d','F6_d','F7_d','F8_d','F9_d','F10_d','F11_d','F12_d','F0_dd','F1_dd','F2_dd','F3_dd','F4_dd','F5_dd','F6_dd','F7_dd','F8_dd','F9_dd','F10_dd','F11_dd','F12_dd','F0','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12'])]

val_set_y_2_1 = encoder(val_set_y_2)

val_set_Inhale_count2 = np.count_nonzero(val_set_y_2_1 == 0)

val_set_Exhale_count2 = np.count_nonzero(val_set_y_2_1 == 1)

val_set_x_2 = val_set_mfcc2.loc[:, val_set_mfcc2.columns.drop(['Name','Unnamed: 0','phon_idx','phon'])]    

val_set_x_2_1 = reshaper(val_set_x_2)

###### Test Set Creation #####

test_set_mfcc2 = []

for k in test_set2:
   
    test_sub_mfcc_dir2 = os.path.join(test_set_dir2, k)
   
    test_sub_mfcc2 = pd.read_csv(test_sub_mfcc_dir2, sep = ',')
   
    test_set_mfcc2.append(test_sub_mfcc2)

test_set_mfcc2 = pd.concat(test_set_mfcc2)

#test_set_mfcc2 = test_set_mfcc2.sample(frac = 1)

test_set_y_2 = test_set_mfcc2.loc[:, test_set_mfcc2.columns.drop(['Name','Unnamed: 0','phon_idx','F0_d','F1_d','F2_d','F3_d','F4_d','F5_d','F6_d','F7_d','F8_d','F9_d','F10_d','F11_d','F12_d','F0_dd','F1_dd','F2_dd','F3_dd','F4_dd','F5_dd','F6_dd','F7_dd','F8_dd','F9_dd','F10_dd','F11_dd','F12_dd','F0','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12'])]

test_set_y_2_1 = encoder(test_set_y_2)

test_set_Inhale_count2 = np.count_nonzero(test_set_y_2_1 == 0)

test_set_Exhale_count2 = np.count_nonzero(test_set_y_2_1 == 1)

test_set_x_2 = test_set_mfcc2.loc[:, test_set_mfcc2.columns.drop(['Name','Unnamed: 0','phon_idx','phon'])]    

test_set_x_2_1 = reshaper(test_set_x_2)    

val2_overall=[]
y2_overall=[]


for i in range(10):
    model = Sequential()
    model.add(Conv1D(filters = 13 ,kernel_size = 3,strides=1,padding='same', input_shape = (train_set_x_2_1.shape[1], 1 ) , activation="relu"))
    model.add(MaxPooling1D())
    model.add(BatchNormalization())
    #model.add(Activation('tanh'))
    model.add(LSTM(80, input_shape = (train_set_x_2_1.shape[1] ,1  ), return_sequences=True))
    #model.add(Conv1D(filters = 13, kernel_size = 3))
    #model.add(Dense(100, activation = 'relu'))
    #model.add(LSTM(100, input_shape = (train_set_x_2_1.shape[1] ,1  ), return_sequences=True))
    #model.add(LSTM(3, return_sequences=False))
    #model.add(TimeDistributed(Dense(64, activation='tanh')))
    model.add(Dropout(0.2))
        #model.add(LSTM(5, input_shape = (setx_1_train_1.shape[1] ,1  ), return_sequences=True))
        #model.add(BatchNormalization())
    model.add(Dense(80, activation = 'relu'))
    #model.add(Conv1D(filters = 13, kernel_size = 12))
        #model.add(Dropout(0.2))
    #model.add(Dense(900, activation = 'relu'))
        #model.add(Dropout(0.4))
    model.add(Flatten())
        #model.add(Dropout(0.4))
    model.add(Dense(1 ,activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005) , metrics=['accuracy'])
    model.summary()
    
    es2 = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=40)

    initial_weights2 = model.get_weights()

        ########### Model Fit Fold2 ##################
    #model.compile(loss='binary_crossentropy', optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005) , metrics=['accuracy'])
    history_set2 = model.fit(train_set_x_2_1,train_set_y_2_1, batch_size=2056 , epochs = 200 , shuffle = True,
                            validation_data=(val_set_x_2_1, val_set_y_2_1) ,callbacks=[es2])

    final_weights2 = model.get_weights()
    plotter(history_set2)

    val2 = model.evaluate(test_set_x_2_1,test_set_y_2_1)

    y_pred2= (model.predict(test_set_x_2_1) > 0.5).astype("int32")
    y_actu2=(test_set_y_2_1)
    cf2=confusion_matrix(y_actu2, y_pred2)

    plot_confusion1(cf2)
    print("F1 SCORE OF 1ST FOLD")
    print(f1_score(y_actu2, y_pred2))
    y2=f1_score(y_actu2, y_pred2)
     
    val2_overall.append(val2[1])
    y2_overall.append(y2)


print(val2_overall)
print(y2_overall)

val_mean = np.mean(val2_overall)

y_mean = np.mean(y2_overall)

print("Averege Test Accuracy = ", val_mean)

print("Average F1 Score = " , y_mean)

    




'''
model_m = Sequential()
#model_m.add(Reshape((TIME_PERIODS, num_sensors), input_shape=(input_shape,)))
model_m.add(Conv1D(filters = 13 ,kernel_size = 12,strides=1,padding='same', input_shape = (train_set_x_2_1.shape[1], 1 ) , activation="relu"))
model_m.add(Conv1D(100, 20, activation='relu'))
model_m.add(Conv1D(160, 20, activation='relu'))
#model_m.add(MaxPooling1D(3))
#model_m.add(Conv1D(16, 20, activation='relu'))
#model_m.add(Conv1D(16, 20, activation='relu'))
model_m.add(GlobalAveragePooling1D())
model_m.add(Dropout(0.2))
model_m.add(Dense(1, activation='softmax'))
print(model_m.summary())



# serialize model to JSON
model_json = model.to_json()
with open("model_5.json", "w") as json_file:
     json_file.write(model_json)
# # serialize weights to HDF5
model.save_weights("model_5.h5")
print("Saved model to disk")

jsonString = json.dumps(val2)
jsonFile = open("data.json", "w")
jsonFile.write(jsonString)
jsonFile.close()

'''