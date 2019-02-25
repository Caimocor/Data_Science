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
