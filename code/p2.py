# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 10:24:51 2019

@author: if715029
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import sklearn.metrics as skm
import scipy.spatial.distance as sc 

#%%

digits = datasets.load_digits()

#%%
ndig = 100
for k in np.arange(ndig):
    plt.subplot(np.floor(np.sqrt(ndig)),np.ceil(np.sqrt(ndig)),k+1)
    plt.axis('off')
    plt.imshow(digits.images[k],cmap=plt.cm.gray_r)
    
#%% Seleccionar una data de los datos y convertirlos a binarios
    
# Emparejamiento Simple y Jaccard son únicamente para binarios. 
# El set predefinido contiene dígitos de 0 a 15, se cambiarán a 0's y 1's

data = digits['data'][0:30]

umbral = 7
data[data<=umbral] = 0
data[data>umbral] = 1

data = pd.DataFrame(data)

#%% Calcular indices de similitud

cf_m = skm.confusion_matrix(data.iloc[0,:],data.iloc[10,:]) # Matriz de confusión. 

sim_simple = skm.accuracy_score(data.iloc[0,:],data.iloc[10,:])
sim_simple_manual = (cf_m[0,0]+cf_m[1,1])/np.sum(cf_m)

#sim_jac = skm.jaccard_similarity_score(data.iloc[0,:],data.iloc[10,:]) # No está habilitado, utiliza el indice de 
sim_jac_manual = cf_m[1,1]/(np.sum(cf_m)-cf_m[0,0])
#%%

d1 = sc.hamming(data.iloc[0,:],data.iloc[10,:]) #distancia de emparejamiento simple (1-emparejamiento)
d2 = sc.jaccard(data.iloc[0,:],data.iloc[10,:]) #distancia de emparejamiento de Jaccard. 

#%% Calcular todas las combinaciones posibles. 

D1 = sc.pdist(data,'hamming')
D1 = sc.squareform(D1)

D2 = sc.pdist(data,'jaccard')
D2 = sc.squareform(D2)
