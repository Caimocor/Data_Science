# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 18:37:21 2019

@author: 8531313
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 09:03:24 2019
@author: snc6420
"""

import numpy as np
import pandas as pd
from random import randint
import math
from sklearn.neighbors import KNeighborsClassifier
import random
import time
import multiprocessing as mp
    

    
    


#cria as colunas
colmns = []
for i in range(1,65):
    colmns.append('col_'+str(i))
    
#cria o dataframe com as colunas desejadas    
df = pd.DataFrame(columns=colmns)
tam = 1000
#cria arrays aleatórios
for c in df.columns:
    
    #cria a quantidade de nulos
    qt_nulos = random.random()
    df[c] = np.random.choice([0, 1], size=tam, p=[qt_nulos, 1-qt_nulos])

qt_nan = tam*0.008
for col in df.columns:
    #gera a quantidade de NaN que serão colocados no DF
    num = int(qt_nan)
    #gera as coluna e linhas aleatórias que receberão os NaNs
    for n in range(0,num):
        numbr = randint(0, tam - 1)
        df[col][numbr] = np.nan
        
def knn_fill_missing(df):   
    columns_names = df.columns     
    #realiza um for para observar cada linha
    for row in df.itertuples():

        #array que possui o nome da coluna com valor NaN em cada observação
        col_nan_names = []

        qt_nan = 0
        #faz um for para olhar cada coluna
        for col_ind in range(0,len(columns_names)):

            x = row[1:]
            x = x[col_ind]
            
            #verifica se o valor daquela linha e coluna específca é um NaN
            if math.isnan(x):
                #adiciona a quantidade de nan presentes em uma observação
                qt_nan += qt_nan
                
                #adiciona o nome da coluna com nan
                col_nan_names.append(columns_names[col_ind])
                
        
        #se a quantidade de valores nulos numa linha for igual ao numero de colunas menos 1
        if qt_nan >= len(columns_names) - 1:
            
            #retira aquela observação do df_bool
            df.drop(row[0])
            
            #segue para a proxima linha
            continue
        
    
        for r in range(0,len(col_nan_names)):
                df_knn = df.dropna()
                X = df_knn.drop(columns=col_nan_names[:len(col_nan_names)-r])
                
                y = df_knn[col_nan_names[r]]
                
                KNN = KNeighborsClassifier(algorithm = 'kd_tree',n_neighbors=3, n_jobs = -1)
                KNN.fit(X, y)
                
                #salva a obsevação numa variável
                inp =df.loc[row[0]].values
                inp = inp[~np.isnan(inp)]
                df[col_nan_names[r]][row[0]] = KNN.predict(inp.reshape(1, -1))

        
    return df

#a ultima coluna nao está sendo alterada pelo KNN portanto deverá ser alterada

df[df.columns[-1]] = df[df.columns[-1]]
start_time = time.time()
df = knn_fill_missing(df)
print("--- %s seconds ---" % (time.time() - start_time))
print('Quantidade de nulos depois')
print(df.isnull().sum().sum())
