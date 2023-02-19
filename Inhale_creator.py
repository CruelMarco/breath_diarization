# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 16:31:04 2023

@author: Spirelab
"""



import numpy as np
import soundfile as sf
from tqdm import tqdm
import os
import librosa
import pandas as pd
from pandas import DataFrame as df

#os.chdir('C:/Users/Spirelab/Desktop/Breath_gender/Shivani_data')

dir = 'C:/Users/Spirelab/Desktop/Breath_Diarization/test'

os.chdir(dir)

mfcc_store_dir = 'C:/Users/Spirelab/Desktop/Breath_Diarization/test_mfcc'

files = os.listdir(dir)

wav_files = [f for f in files if f.endswith(".wav")]

txt_files = [f for f in files if f.endswith(".txt")]

male_count = 0

female_count = 0
    
############# MFCC Calculation ###############

for j in tqdm(wav_files) :
    
    audio_path = os.path.join(dir, j)
    
    audio_file, fs = librosa.load(audio_path, sr = 16000, mono = True)
    
    annot_path = audio_path[0 : -3] + 'txt'
    
    annot_file = pd.read_csv(annot_path , sep = "\t", names = ['start', 'end', 'phon'] , header = None)
    
    gender = j.split("_")[8]
    
    if gender == 'M' :
        
        male_count+=1
    else :
        
        female_count+=1
    
    name = j.split("_")[5]
    
    phon_col = annot_file['phon']
    
    st_col = annot_file['start']
    
    end_col = annot_file['end']
    
    ###Inhale DF###
    
    ii_idx = [i for i in range(len(phon_col)) if "ii" in phon_col[i]]
    
    ii_st_idx = st_col[ii_idx]
    
    ii_end_idx = end_col[ii_idx]
    
    ii_st_sam = list(np.ceil(ii_st_idx * fs))
    
    ii_end_sam = list(np.ceil(ii_end_idx*fs))
    
    #ii_st_sam = [int(i) for i in ii_st_sam]
    
    ii_chunk = []
    
    ii_chunk_mfcc_df = []
    
    sub_ii_mfcc_df = []
    
    #print(j)
    
    for k in range(len(ii_st_sam)) :
        
        ii_chunk = audio_file[int(ii_st_sam[k]) : int(ii_end_sam[k])]
        
        mfcc_ii = librosa.feature.mfcc(ii_chunk , sr = fs , n_mfcc = 13 , win_length = 320 , hop_length = 160)
                
        ii_chunk_mfcc_df = np.array(np.transpose(mfcc_ii))
        
        delta = np.array(np.transpose(librosa.feature.delta(mfcc_ii)))
        
        delta_ii_chunk_mfcc_df = pd.DataFrame(delta , columns = ['F0_d' , 'F1_d' , 'F2_d' , 'F3_d' , 'F4_d' , 'F5_d' , 'F6_d' , 'F7_d' , 'F8_d' , 'F9_d' , 'F10_d' , 'F11_d' , 'F12_d'])

        delta2 = np.array(np.transpose(librosa.feature.delta(mfcc_ii, order=2)))
        
        delta2_ii_chunk_mfcc_df = pd.DataFrame(delta2 , columns = ['F0_dd' , 'F1_dd' , 'F2_dd' , 'F3_dd' , 'F4_dd' , 'F5_dd' , 'F6_dd' , 'F7_dd' , 'F8_dd' , 'F9_dd' , 'F10_dd' , 'F11_dd' , 'F12_dd'])
        
        ii_chunk_mfcc_df = pd.DataFrame(ii_chunk_mfcc_df , columns = ['F0' , 'F1' , 'F2' , 'F3' , 'F4' , 'F5' , 'F6' , 'F7' , 'F8' , 'F9' , 'F10' , 'F11' , 'F12'])
        
        index_df = pd.DataFrame([ii_idx[k] ]*ii_chunk_mfcc_df.shape[0], columns = ['phon_idx'])
        
        mfcc_ii_delta_delta2 = pd.concat([index_df , ii_chunk_mfcc_df, delta_ii_chunk_mfcc_df, delta2_ii_chunk_mfcc_df],axis=1, join='inner')
        
        #print(mfcc_ii_delta_delta2)
        
        sub_ii_mfcc_df.append(mfcc_ii_delta_delta2)
        
    sub_ii_mfcc_df = pd.concat(sub_ii_mfcc_df, ignore_index=True)
     
    sub_ii_mfcc_df.insert(0,'Name' , name , True)
    
    phon_ii = "ii"
    
    sub_ii_mfcc_df['phon'] = phon_ii
    
    
    
    
    
    