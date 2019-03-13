# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 09:31:25 2019

@author: if715029, Josean
"""

import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader.data as web
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.cluster import KMeans

#%% Descargar los datos de la serie de tiempo
inicio = datetime(2018,3,12)
final = datetime(2019,3,12)
data = web.YahooDailyReader(symbols='NFLX',start=inicio,end=final,interval='d').read()#d=diario

#%% Visualizar al serie de datos
plt.plot(data['Close'])
plt.show()

#%% Descomponer la serie original en subseries de longitud nv
dat = data['Close']
nv = 5 #num. de columnas a crear
n_prices = len(dat) #cuenta cuantos datos (252)
dat_new = np.zeros((n_prices-nv+1,nv)) #generar la matriz con ceros
for k in np.arange(nv):
    dat_new[:,k] = dat[k:(n_prices-nv+1)+k] #incluir los ultimos 5 datos en la matriz

#%%Normalizar los datos
tmp = dat_new.transpose()
tmp = (tmp-tmp.mean(axis=0))/tmp.std(axis=0)    
dat_new = tmp.transpose()

#%% Ver las ventanas
plt.plot(dat_new.T)
plt.xlabel('time')
plt.ylabel('prices')
plt.show()

#%% Buscar patrones con el algoritmo de clustering
n_clusters = 15
inercias = np.zeros(n_clusters)
for k in np.arange(n_clusters)+1:
    model = KMeans(n_clusters=k,init='k-means++').fit(dat_new)
    inercias[k-1] = model.inertia_

plt.plot(np.arange(n_clusters)+1,inercias) #plt.plot(x,y)
plt.xlabel('# grupos')
plt.ylabel('inercias')
plt.show()

#%% Clasificar datos segun el codo y graficar los centroides
model = KMeans(n_clusters=4,init='k-means++').fit(dat_new)
grupos = model.predict(dat_new)
centroides = model.cluster_centers_
plt.plot(centroides.T)
plt.grid()
plt.show()

#%% Dibujar los centroides por separado
n_subfig = np.ceil(np.sqrt(len(np.unique(grupos))))
for k in np.unique(grupos):
    plt.subplot(n_subfig,n_subfig,k+1)
    plt.plot(centroides[k,:])
    plt.ylabel('grupo %d')
plt.show()

#%% 
plt.subplot(211)
plt.plot(dat)
plt.xlabel('time')
plt.ylabel('price')
plt.subplot(212)
plt.bar(np.arange(nv,len(dat)+1),grupos)
plt.xlabel('time')
plt.ylabel('price')
plt.show()







