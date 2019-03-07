import numpy as np
import pandas as pd
from random import randint

#cria o dataframe com as colunas desejadas
df = pd.DataFrame(columns=['col_1','col_2','col_3','col_4','col_5','col_6','col_7','col_8',])

#cria arrays aleatórios
df['col_1'] = np.random.choice([0, 1], size=1000, p=[.1, .9])
df['col_2'] = np.random.choice([0, 1], size=1000, p=[.2, .8])
df['col_3'] = np.random.choice([0, 1], size=1000, p=[.5, .5])
df['col_4'] = np.random.choice([0, 1], size=1000, p=[.7, .3])
df['col_5'] = np.random.choice([0, 1], size=1000, p=[.8, .2])
df['col_6'] = np.random.choice([0, 1], size=1000, p=[.2, .8])
df['col_7'] = np.random.choice([0, 1], size=1000, p=[.3, .7])
df['col_8'] = np.random.choice([0, 1], size=1000, p=[.4, .6])

for col in df.columns:
    #gera um numero aleatorio entre 0 e 999
    num = randint(0, 100)
    
    for n in range(0,num):
        numbr = randint(0, 999)
        df[col][numbr] = np.nan

idx, idy = np.where(pd.isnull(df))
result = np.column_stack([df.index[idx], df.columns[idy]])



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


#cria o dataframe com as colunas desejadas
df = pd.DataFrame(columns=['col_1','col_2','col_3','col_4','col_5','col_6','col_7','col_8',])
tam = 100
#cria arrays aleatórios
df['col_1'] = np.random.choice([0, 1], size=tam, p=[.1, .9])
df['col_2'] = np.random.choice([0, 1], size=tam, p=[.2, .8])
df['col_3'] = np.random.choice([0, 1], size=tam, p=[.5, .5])
df['col_4'] = np.random.choice([0, 1], size=tam, p=[.7, .3])
df['col_5'] = np.random.choice([0, 1], size=tam, p=[.8, .2])
df['col_6'] = np.random.choice([0, 1], size=tam, p=[.2, .8])
df['col_7'] = np.random.choice([0, 1], size=tam, p=[.3, .7])
df['col_8'] = np.random.choice([0, 1], size=tam, p=[.4, .6])

for col in df.columns:
    #gera a quantidade de NaN que serão colocados no DF
    num = randint(0, tam/10)
    #gera as coluna e linhas aleatórias que receberão os NaNs
    for n in range(0,num):
        numbr = randint(0, tam - 1)
        df[col][numbr] = np.nan
        


#idx, idy = np.where(pd.isnull(df))
#result = np.column_stack([df.index[idx], df.columns[idy]])
#print(result)



print('Quantidade de nulos antes')
print(df.isna().sum().sum())
################# Inicio do KNN #######################################

#essa função recebe um dataframe com valores booleanos e realiza um KNN para prever o possível valor
def knn_fill_missing(df):   
    columns_names = df.columns     
    #realiza um for para observar cada linha
    for row in df.iterrows():
        print(row[0])
        #array que possui o nome da coluna com valor NaN em cada observação
        col_nan_names = []
        print(row)
        qt_nan = 0
        #faz um for para olhar cada coluna
        for col_ind in range(1,len(columns_names)):
            print('------------')
            x = row[1]
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
                print('VALOR DO R',r)
                df_knn = df.dropna()
                print(col_nan_names[:len(col_nan_names)-r])
                X = df_knn.drop(columns=col_nan_names[:len(col_nan_names)-r])
                
                print('QUANTIDADE DE COLUNAS EM X: ',len(X.columns))
                y = df_knn[col_nan_names[r]]
                
                KNN = KNeighborsClassifier(algorithm = 'kd_tree',n_neighbors=3, n_jobs = -1)

                KNN.fit(X, y)
                
                #salva a obsevação numa variável
                inp =df.loc[row[0]].values
                print('QUANTIDADE DE VALORES QUE ENTRARÃO NO PREDICT antes da exlusão de nan: ',inp)
                inp = inp[~np.isnan(inp)]
                print('QUANTIDADE DE VALORES QUE ENTRARÃO NO PREDICT: ',inp)
                print('Printa o que possui na variáveis antes de ser modificada:',df[col_nan_names[r]][row[0]])
                df[col_nan_names[r]][row[0]] = KNN.predict(inp.reshape(1, -1))
                #retira o valor nulo da mesma
                #inp = inp[~np.isnan(inp)]
                #inp = inp.reshape(1, -1)
                #inp = inp[0]
                #print(inp)
                #substitui a linha e coluna que possui NaN pelo valor específico
                #df[col_nan_names[r]][row[0]] = KNN.predict([inp])
        
    return df

#a ultima coluna nao está sendo alterada pelo KNN portanto deverá ser alterada

df['col_8'] = df['col_8']
df = knn_fill_missing(df)
print('Quantidade de nulos depois')
print(df.isnull().sum().sum())
print(df)
