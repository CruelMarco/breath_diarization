#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 15:37:21 2023

@author: shaique
"""

import pandas as pd
import os
from tqdm import tqdm

dir = 'C:/Users/Spirelab/Desktop/Breath_Diarization/mfcc_d_dd_ii_xx/test_mfcc'

ii_dir = 'C:/Users/Spirelab/Desktop/Breath_Diarization/mfcc_d_dd_ii_xx/mfcc_ii_d_dd'

xx_dir = 'C:/Users/Spirelab/Desktop/Breath_Diarization/mfcc_d_dd_ii_xx/mfcc_xx_d_dd'

os.chdir(dir)

files = os.listdir(dir)

for i in tqdm(files): 
    
    #print(i)

    csv_dir = os.path.join(dir, i)

    df = pd.read_csv(csv_dir, sep=',')

    df = df.drop(['Unnamed: 0'], axis=1)

    pho_idx = list(df['phon_idx'])

    unique_idx = []
    
    ii_df = []
    
    xx_df = []
    
    ii_size_df = []
    
    xx_size_df = []


    [unique_idx.append(x) for x in pho_idx if x not in unique_idx]
    
    for j in unique_idx:
        
        
        #print(j)
        
        if j % 2 == 0:
            
            df_j_ii = df[df['phon_idx'] == j]
            
            df_name_ii = i[0 : -13] + 'ii_' + str(j) + '.csv'
            
            ii_size = len(df_j_ii)
            
            phon = 'ii'
            
            ii_size_df_2 = [['file_name' , i] , ['index' , j] , ['phon' ,  phon] , ['size' , ii_size]]
            
            ii_size_df = pd.DataFrame(ii_size_df_2, columns = ['file_name' , 'index', 'phon' , 'size'])
            
            
            
            #ii_csv_dir = os.path.join(ii_dir , df_name_ii)
            
            #df_j_ii.to_csv(ii_csv_dir)
            
            #pd.to_csv()
            
            #ii_df.append(df_j_ii)
            
            
            
            #df_j_ii = df_j_ii.set_index('phon', append=True).swaplevel(0,1)

            
        else :
            
            df_j_xx = df[df['phon_idx'] == j]
            
            df_name_xx = i[0 : -13] + 'xx_' + str(j) + '.csv'
            
            xx_size = len(df_j_xx)
            
            phon = 'xx'

            xx_size_df = [['file_name' , i] , ['index' ,j] , ['phon' ,  phon] , ['size' , xx_size]]
            
            xx_size_df = pd.DataFrame(xx_size_df, columns = ['file_name' , 'index', 'phon' , 'size'])
            
            #xx_csv_dir = os.path.join(xx_dir , df_name_xx)
            
            #df_j_xx.to_csv(xx_csv_dir)
            
            #xx_df.append(df_j_xx)
            
    
 
    # for j in unique_idx:
        
    #     if j % 2 == 0:
        
