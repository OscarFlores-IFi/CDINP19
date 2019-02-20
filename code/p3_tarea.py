#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 19:53:47 2019

@author: fh
"""

import pandas as pd
import numpy as np
import sklearn.metrics as skm
import scipy.spatial.distance as sc

#%% Leer datos
data = pd.read_excel('../data/Test de películas(1-16).xlsx', encoding='latin_1')

#%% Tomar solamente las columnas con calificación de películas. 
pel = pd.DataFrame()
for i in range((len(data.T)-5)//3):
    pel = pel.append(data.iloc[:,6+i*3])
pel = pel.T
print(pel)

#%% convertir a binarios
pel_n = pel.copy()
pel_n[pel_n<3] = 0
pel_n[pel_n>=3] = 1

#%% calcular Jaccard para los usuarios. 
D1 = sc.squareform(sc.pdist(pel_n,'jaccard'))

Isim1 = 1-D1

#%% Elegir usuario y tomar los usuarios más parecidos a él. 
user = 1
Isim_user = Isim1[user]
indx_user1 = np.argsort(Isim_user)
 
#%% Recomendar las películas que el usuario no ha visto, pero el recomendador sí. 
USER = pel_n.loc[user]
USER_sim1 = pel_n.loc[indx_user1[-2]]

indx_recomend1_1 = (USER_sim1==1)&(USER==0)
recomend1_1 = list(USER.index[indx_recomend1_1])

#%% Recomendar por segundo método
USER_sim1 = np.mean(pel_n.loc[indx_user1[-6:-1]],axis = 0)
USER_sim1[USER_sim1<=.5]=0
USER_sim1[USER_sim1>.5]=1

indx_recomend1_2 = (USER_sim1==1)&(USER==0)
recomend1_2 = list(USER.index[indx_recomend1_2])










#%% Ahora con Dummies. 
pel_d = pd.get_dummies(pel[pel.columns[0]],prefix=pel.columns[0])
for i in pel.columns[1:]:
    tmp = pd.get_dummies(pel[i],prefix=i)
    pel_d = pel_d.join(tmp)

#%% Jaccard
D2 = sc.squareform(sc.pdist(pel_d,'jaccard'))

Isim2 = 1-D2

#%%
indx_user2 = np.argsort(Isim_user)

#%%
USER = pel_d.loc[user]
USER_sim2 = pel_d.loc[indx_user2[-2]]

indx_recomend2_1 = (USER_sim2==1)&(USER==0)
recomend2_1 = list(USER.index[indx_recomend2_1])

#%%
USER_sim2 = np.mean(pel_d.loc[indx_user2[-6:-1]],axis = 0)
USER_sim2[USER_sim2<=.5]=0
USER_sim2[USER_sim2>.5]=1

indx_recomend2_2 = (USER_sim2==1)&(USER==0)
recomend2_2 = list(USER.index[indx_recomend2_2])









