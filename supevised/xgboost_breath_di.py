# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 16:13:17 2022

@author: Ashutosh
"""

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from tqdm import tqdm
import scipy.signal
import os
import librosa
import pandas as pd
from pandas import DataFrame as df
import glob, os
import json
import sklearn
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
from sklearn.metrics import accuracy_score
from numpy import loadtxt
import xgboost as xgb
from sklearn import preprocessing
###### For Hyperparameter tuning ########
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe


#from sklearn import 

dir = '/data/breath_diarization_datasets/mfcc_datasets/mfcc_iixx_d_dd_2/sets'

os.chdir(dir)

sets = os.listdir(dir)

label_encoder = preprocessing.LabelEncoder()

set_path = []

for i in sets:
    
    set_dir = os.path.join(dir,i)
    
    set_path.append(set_dir)
    
set_files = []

for j in set_path:
    
    files = os.listdir(j)
    
    set_files.append(files)
    
list_of_path=[]

for z in range (len(set_path)):
    
    #set_no=os.listdir(ee(z))
    set=set_files[z]
    
    for j in range (len(set_files[z])):
        
        filepath=set_path[z]+'/'+set_files[z][j]
        
        list_of_path.append(filepath)
        
set1=list(filter(lambda k: 'set1' in k, list_of_path))

set2=list(filter(lambda k: 'set2' in k, list_of_path))

set3=list(filter(lambda k: 'set3' in k, list_of_path))

set4=list(filter(lambda k: 'set4' in k, list_of_path))

set5=list(filter(lambda k: 'set5' in k, list_of_path))

value1=[]

for n in range (len(set1)):
    
    table1=pd.read_csv(set1[n],index_col=0)
    
    value1.append(table1)

value2=[]

for n in range (len(set2)):
    
    table2=pd.read_csv(set2[n],index_col=0)
    
    value2.append(table2)

value3=[]

for n in range (len(set3)):
    
    table3=pd.read_csv(set3[n],index_col=0)
    
    value3.append(table3)

value4=[]

for n in range (len(set4)):
    
    table4=pd.read_csv(set4[n],index_col=0)
    
    value4.append(table4)

value5=[]

for n in range (len(set5)):
    
    table5=pd.read_csv(set5[n],index_col=0)
    
    value5.append(table5)

mfcc1=pd.concat(value1)

mfcc2=pd.concat(value2)

mfcc3=pd.concat(value3)

mfcc4=pd.concat(value4)

mfcc5=pd.concat(value5)

mfcc1=mfcc1.sample(frac=1)

mfcc2=mfcc2.sample(frac=1)

mfcc3=mfcc3.sample(frac=1)

mfcc4=mfcc4.sample(frac=1)

mfcc5=mfcc5.sample(frac=1)

trainx_1 = mfcc1.loc[:, mfcc1.columns.drop(['Name','phon_idx','phon'])]

trainy_1 = mfcc1.loc[:, mfcc1.columns.drop(['Name','phon_idx','F0_d','F1_d','F2_d','F3_d','F4_d','F5_d','F6_d','F7_d','F8_d','F9_d','F10_d','F11_d','F12_d','F0_dd','F1_dd','F2_dd','F3_dd','F4_dd','F5_dd','F6_dd','F7_dd','F8_dd','F9_dd','F10_dd','F11_dd','F12_dd','F0','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12'])]

trainy_1["phon"] = label_encoder.fit_transform(trainy_1["phon"])

trainx_2 = mfcc2.loc[:, mfcc2.columns.drop(['Name','phon_idx','phon'])]

trainy_2 = mfcc2.loc[:, mfcc2.columns.drop(['Name','phon_idx','F0_d','F1_d','F2_d','F3_d','F4_d','F5_d','F6_d','F7_d','F8_d','F9_d','F10_d','F11_d','F12_d','F0_dd','F1_dd','F2_dd','F3_dd','F4_dd','F5_dd','F6_dd','F7_dd','F8_dd','F9_dd','F10_dd','F11_dd','F12_dd','F0','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12'])]

trainy_2["phon"] = label_encoder.fit_transform(trainy_2["phon"])

trainx_3 = mfcc3.loc[:, mfcc3.columns.drop(['Name','phon_idx','phon'])]

trainy_3 = mfcc3.loc[:, mfcc3.columns.drop(['Name','phon_idx','F0_d','F1_d','F2_d','F3_d','F4_d','F5_d','F6_d','F7_d','F8_d','F9_d','F10_d','F11_d','F12_d','F0_dd','F1_dd','F2_dd','F3_dd','F4_dd','F5_dd','F6_dd','F7_dd','F8_dd','F9_dd','F10_dd','F11_dd','F12_dd','F0','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12'])]

trainy_3["phon"] = label_encoder.fit_transform(trainy_3["phon"])

trainx_4 = mfcc4.loc[:, mfcc4.columns.drop(['Name','phon_idx','phon'])]

trainy_4 = mfcc4.loc[:, mfcc4.columns.drop(['Name','phon_idx','F0_d','F1_d','F2_d','F3_d','F4_d','F5_d','F6_d','F7_d','F8_d','F9_d','F10_d','F11_d','F12_d','F0_dd','F1_dd','F2_dd','F3_dd','F4_dd','F5_dd','F6_dd','F7_dd','F8_dd','F9_dd','F10_dd','F11_dd','F12_dd','F0','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12'])]

trainy_4["phon"] = label_encoder.fit_transform(trainy_4["phon"])

trainx_5 = mfcc5.loc[:, mfcc5.columns.drop(['Name','phon_idx','phon'])]

trainy_5 = mfcc5.loc[:, mfcc5.columns.drop(['Name','phon_idx','F0_d','F1_d','F2_d','F3_d','F4_d','F5_d','F6_d','F7_d','F8_d','F9_d','F10_d','F11_d','F12_d','F0_dd','F1_dd','F2_dd','F3_dd','F4_dd','F5_dd','F6_dd','F7_dd','F8_dd','F9_dd','F10_dd','F11_dd','F12_dd','F0','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12'])]

trainy_5["phon"] = label_encoder.fit_transform(trainy_5["phon"])


setx_1_train= pd.concat([trainx_2,trainx_3,trainx_4,trainx_5])

setx_2_train= pd.concat([trainx_1,trainx_3,trainx_4,trainx_5])

setx_3_train= pd.concat([trainx_1,trainx_2,trainx_4,trainx_5])

setx_4_train= pd.concat([trainx_1,trainx_2,trainx_3,trainx_5])

setx_5_train= pd.concat([trainx_1,trainx_2,trainx_3,trainx_4])

setx_1_test= trainx_1

setx_2_test= trainx_2

setx_3_test= trainx_3

setx_4_test= trainx_4

setx_5_test= trainx_5

sety_1_train= pd.concat([trainy_2,trainy_3,trainy_4,trainy_5])

sety_2_train= pd.concat([trainy_1,trainy_3,trainy_4,trainy_5])

sety_3_train= pd.concat([trainy_1,trainy_2,trainy_4,trainy_5])

sety_4_train= pd.concat([trainy_1,trainy_2,trainy_3,trainy_5])

sety_5_train= pd.concat([trainy_1,trainy_2,trainy_3,trainy_4])

sety_1_test= trainy_1

sety_2_test= trainy_2

sety_3_test= trainy_3

sety_4_test= trainy_4

sety_5_test= trainy_5


############# Model Fitting ###########



space={'max_depth': hp.quniform("max_depth", 3, 18, 1),
        'gamma': hp.uniform ('gamma', 1,9),
        'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
        'reg_lambda' : hp.uniform('reg_lambda', 0,1),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
        'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
        'n_estimators': 180,
        'seed': 0
    }




classifier = XGBClassifier()

#############Fold 1##############
def objective(space):
    clf=xgb.XGBClassifier(
                    n_estimators =space['n_estimators'], max_depth = int(space['max_depth']), gamma = space['gamma'],
                    reg_alpha = int(space['reg_alpha']),min_child_weight=int(space['min_child_weight']),
                    colsample_bytree=int(space['colsample_bytree']))
    
    evaluation = [( setx_1_train, sety_1_train), ( setx_1_test, sety_1_test)]
    
    clf.fit(setx_1_train, sety_1_train,
            eval_set=evaluation, eval_metric="auc",
            early_stopping_rounds=10,verbose=False)
    

    pred = clf.predict(setx_1_test)
    accuracy = accuracy_score(sety_1_test, pred>0.5)
    print ("SCORE:", accuracy)
    return {'loss': -accuracy, 'status': STATUS_OK }



trials = Trials()

best_hyperparams = fmin(fn = objective,
                        space = space,
                        algo = tpe.suggest,
                        max_evals = 100,
                        trials = trials)


classifier = XGBClassifier(params = best_hyperparams)

fold1_fit = classifier.fit(setx_1_train, sety_1_train)

y_pred_fold_1 = classifier.predict(setx_1_test)

score_fold_1_test=fold1_fit.score(setx_1_test, sety_1_test)

score_fold_1_train = fold1_fit.score(setx_1_train, sety_1_train)

###########Fold 2################

fold2_fit = classifier.fit(setx_2_train, sety_2_train)

y_pred_fold_2 = classifier.predict(setx_2_test)

score_fold_2_test = fold2_fit.score(setx_2_test, sety_2_test)

score_fold_2_train = fold2_fit.score(setx_2_train, sety_2_train)



#accuracies2 = cross_val_score(estimator = classifier2, X = setx_2_train, y = sety_2_train, cv = None)

###########Fold 3################

fold3_fit = classifier.fit(setx_3_train, sety_3_train)

y_pred_fold_3 = classifier.predict(setx_2_test)

score_fold_3_test = fold3_fit.score(setx_3_test, sety_3_test)

score_fold_3_train = fold3_fit.score(setx_3_train, sety_3_train)

###########Fold 4################

fold4_fit = classifier.fit(setx_4_train, sety_4_train)

y_pred_fold_4 = classifier.predict(setx_4_test)

score_fold_4_test = fold4_fit.score(setx_4_test, sety_4_test)

score_fold_4_train = fold4_fit.score(setx_4_train, sety_4_train)


###########Fold 5################

fold5_fit = classifier.fit(setx_5_train, sety_5_train)

y_pred_fold_5 = classifier.predict(setx_5_test)

score_fold_5_test = fold5_fit.score(setx_5_test, sety_5_test)

score_fold_5_train = fold5_fit.score(setx_5_train, sety_5_train)

print("Test Score is = " , (score_fold_1_test+score_fold_2_test+score_fold_3_test+score_fold_4_test+score_fold_5_test)/5)

print("Train Score is = " , (score_fold_1_train+score_fold_2_train+score_fold_3_train+score_fold_4_train+score_fold_5_train)/5)


