# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 09:17:19 2019

@author: if715029
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as sc
import pandas as pd
from cdin import cdinp19 as cd

#%% Leer datos 
estaciones = ['Atemajac','Aguilas','Centro','Las Pintas','Loma Dorada','Miravalle','Oblatos','Santa Fe','Tlaquepaque','Vallarta']
data = []
for estacion in estaciones:
    tmp = pd.read_excel('../data/contaminacion_2015.xlsx',sheet_name=estacion,index_col=1)
    tmp = tmp.iloc[:,1:6]
    data.append(tmp)

#%% Data Quality Report
dqr = cd.dqr(data[0])

#%% Distancia entre CO's 
Co = np.zeros((8016,len(data)))
for i in range(len(data)):
    Co[:,i] = data[i].CO[:8016].values
Co = pd.DataFrame(Co)
Co = Co.dropna()

#%%
Mat_co = sc.squareform(sc.pdist(Co.T,'euclidean'))

#%%
Pm10 = np.zeros((8016,len(data)))
for i in range(len(data)):
    Pm10[:,i] = data[i].PM10[:8016].values
Pm10 = pd.DataFrame(Pm10)
Pm10 = Pm10.dropna()

#%%
Mat_pm10 = sc.squareform(sc.pdist(Pm10.T,'euclidean'))








