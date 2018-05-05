import string
import csv
import pandas as pd


#retira os semicolons do texto e troca por virgula
s = open("USP.txt",encoding="utf8").read()
s = s.replace(';', ',')

f = open("NOVO_USP.csv", 'w',encoding="utf-8")

f.write(s)

f.close()



csv_f = csv.reader(open('NOVO_USP.csv',encoding="utf8"))

mat = []

for row in csv_f:
            mat.append(row)

data = pd.DataFrame(mat)

#copias colunas com dados importantes para colunas que possuem dados sem significado
data.loc[:,11] = data.loc[:,12]
data.loc[:,12] = data.loc[:,14]

#deleta colunas geradas a mais
del data[18]
del data[17]
del data[16]
del data[15]
del data[14]
del data[13]


data.columns = data.loc[0,:]


data = data.iloc[1:]


data.rename(columns={'Líquido': 'Salario Mensal'}, inplace=True)


data.iloc[:,11] = data.iloc[:,11].astype(float)

data.iloc[:,12] = data.iloc[:,12].astype(float)

data.columns = ["Nome","Unid","Depto","Jornada","Categoria","Classe","Ref/MS","Função","Função de Estrutura","Tempo USP","Parcelas Eventuais","Salario Mensal","Líquido"]

#busca na função pedreiro o maior salario
maior_salario = 0
x = 0
salario = 0
for i in range(len(data)):
    if(data.iloc[i,7] == 'Pedreiro'):
        salario = salario + data.iloc[i,12]
        x = x + 1
        if(data.iloc[i,12] > maior_salario):
            maior_salario = data.iloc[i,12]
            a = data.iloc[i,:]

print(salario/x)

print(maior_salario)

print(a)

###################################################################
#salva num vetor funções todas as denominações de cargos existentes
funçoes = []
for i in  data.iloc[:,7]:
  if i not in funçoes:
    funçoes.append(i)
    

#calcula a media de salarios e o maior salario de cada tipo de função  
maior_salario = 0
x = 0
salario = 0
media_salarios = []
maior_salario_funçoes = []


for i in range(len(funçoes)):
    print(i)
    for j in range(len(data)):
        if(data.iloc[j,7] == funçoes[i]):
            salario = salario + data.iloc[j,12]
            x = x + 1
            if(data.iloc[j,12] > maior_salario):
                maior_salario = data.iloc[j,12]
    maior_salario_funçoes.append(maior_salario)
    media_salarios.append(salario/x)
    maior_salario = 0
    x = 0
    salario = 0

mat1=[]


for e1, e2, e3  in zip(funçoes, media_salarios, maior_salario_funçoes):
    mat1.append(e1)
    mat1.append(e2)
    mat1.append(e3)
    print(e1,e2, e3)

data1 = pd.DataFrame(mat1)
data1.to_csv("Media_salarios.csv")  
