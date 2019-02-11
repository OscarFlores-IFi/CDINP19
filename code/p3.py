# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 09:18:37 2019

@author: if715029
"""

import pandas as pd
import numpy as np
import sklearn.metrics as skm
import scipy.spatial.distance as sc

#%% Leer datos
data = pd.read_excel('../data/Test de películas(1-16).xlsx', encoding='latin_1')

#%% Seleccionar datos (a mi estilo)
pel = pd.DataFrame()
for i in range((len(data.T)-5)//3):
    pel = pel.append(data.iloc[:,6+i*3])
pel = pel.T
print(pel)

#%% Seleccionar datos (estilo Riemann)
csel = np.arange(6,243,3)
cnames = list(data.columns.values[csel])
datan = data[cnames]

#%% Promedios
movie_prom = datan.mean(axis=0)
user_prom = datan.mean(axis=1)

#%% Calificaciones a binarios (>= 3)
datan = datan.copy()
datan[datan<3] = 0
datan[datan>=3] = 1

#%% Calcular distancias de indices de similitud
D1 = sc.pdist(datan,'hamming') # hamming == matching
D1 = sc.squareform(D1)

#D2 = sc.pdist(data_b,'jaccard') # hamming == matching
#D2 = sc.squareform(D2)

Isim1 = 1-D1
#%% Seleccionar usuario y determinar sus parecidos 
user = 13
Isim_user = Isim1[user]
Isim_user_sort = np.sort(Isim_user)
indx_user = np.argsort(Isim_user)

#%% Recomendación de películas p1.
USER = datan.loc[user]
USER_sim = datan.loc[indx_user[-2]]

indx_recomend1 = (USER_sim==1)&(USER==0)
recomend1 = list(USER.index[indx_recomend1])









