#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 15:23:09 2019

@author: Chelsi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

###############################################################################
############################      Ejercicio1      #############################
###############################################################################
#%% Importar datos
data = pd.read_csv('Archivos Examen 2-20190410/ex2c_1_2.csv',index_col=0)

#%% Grafica de codos
inercias = np.zeros(15)
for k in np.arange(len(inercias))+1:
    model = KMeans(n_clusters=k,init='random')
    model = model.fit(data)
    inercias[k-1]= model.inertia_

plt.plot(np.arange(len(inercias))+1,inercias)
plt.xlabel('# grupos')
plt.ylabel('Inercia Total')
plt.show()

#%%
model = KMeans(n_clusters=5,init='k-means++').fit(data)
grupos = model.predict(data)
centroides = model.cluster_centers_
plt.plot(centroides.T)
plt.grid()
plt.show()

#%%
model = KMeans(n_clusters=3,init='k-means++').fit(data)
grupos = model.predict(data)
centroides = model.cluster_centers_
plt.plot(centroides.T)
plt.grid()
plt.show()

#%%
model = KMeans(n_clusters=2,init='k-means++').fit(data)
grupos = model.predict(data)
centroides = model.cluster_centers_
plt.plot(centroides.T)
plt.grid()
plt.show()

#%% Clasificar los datos segun la grafica de codo
model = KMeans(n_clusters=3)
model = model.fit(data)
grupos = model.predict(data)







###############################################################################
############################      Ejercicio2      #############################
###############################################################################
#%% Importar datos
data = pd.read_csv('Archivos Examen 2-20190410/ex2c_2_2.csv',index_col=0)

#%% PCA
media = data.mean(axis=0)    
data_m = data-media
M_cov = np.cov(data_m,rowvar=False)    
w,v = np.linalg.eig(M_cov)

#%% Decidir numero de variables a reducir
porcentaje = w/np.sum(w)
porcentaje_acum = np.cumsum(porcentaje)

plt.figure()
plt.bar(np.arange(len(porcentaje)),porcentaje)
plt.show()

#%% Proyectar datos en nuevas dimensiones
limite = .91
indx = porcentaje_acum<=limite
componentes = w[indx]
M_trans = v[:,indx]

data_new = np.array(np.matrix(data_m)*np.matrix(M_trans))

#%% Recuperar imagenes de las variables reducidas
data_r = np.array(np.matrix(data_new)*np.matrix(M_trans.transpose()))
data_r = data_r + media.values

#%%
plt.plot(data[0:200])
plt.title('original')
plt.show()

plt.plot(data_r[0:200])
plt.title('recuperado')
plt.show()





###############################################################################
############################      Ejercicio4      #############################
###############################################################################
#%% Importar Datos
data = pd.read_csv('Archivos Examen 2-20190410/ex2c_4.csv',index_col=0)

#%% Encontrar número de clusters por gráfica de codos. 
n_clusters = 15
inercias = np.zeros(n_clusters)
for k in np.arange(n_clusters)+1:
    model = KMeans(n_clusters=k,init='k-means++').fit(data)
    inercias[k-1] = model.inertia_

plt.plot(np.arange(n_clusters)+1,inercias+1) #plt.plot(x,y)
plt.xlabel('# grupos')
plt.ylabel('inercias')
plt.show()

#%% Clasificar datos segun el codo y graficar los centroides
model = KMeans(n_clusters=2,init='k-means++').fit(data)
grupos = model.predict(data)
centroides = model.cluster_centers_
plt.plot(centroides.T)
plt.grid()
plt.show()

#%% PCA en la base de datos
media = data.mean(axis=0)    
data_m = data-media
M_cov = np.cov(data_m,rowvar=False)    
w,v = np.linalg.eig(M_cov)

#%% Decidir numero de variables a reducir
porcentaje = w/np.sum(w)
porcentaje_acum = np.cumsum(porcentaje)

plt.figure()
plt.bar(np.arange(len(porcentaje)),porcentaje)
plt.show()

#%% Proyectar datos en nuevas dimensiones
limite = 1
indx = porcentaje_acum<=limite
componentes = w[indx]
M_trans = v[:,indx]

data_new = np.array(np.matrix(data_m)*np.matrix(M_trans))

#%% Recuperar imagenes de las variables reducidas
data_r = np.array(np.matrix(data_new)*np.matrix(M_trans.transpose()))
data_r = data_r + media.values

#%% Encontrar número de clusters por gráfica de codos para datos normalizados
n_clusters = 15
inercias = np.zeros(n_clusters)
for k in np.arange(n_clusters)+1:
    model = KMeans(n_clusters=k,init='k-means++').fit(data_r)
    inercias[k-1] = model.inertia_

plt.plot(np.arange(n_clusters)+1,inercias+1)
plt.xlabel('# grupos')
plt.ylabel('inercias')
plt.show()

#%% Clasificar datos segun el codo y graficar los centroides para datos normalizados. 
model = KMeans(n_clusters=2,init='k-means++').fit(data_r)
grupos = model.predict(data_r)
centroides = model.cluster_centers_
plt.plot(centroides.T)
plt.grid()
plt.show()








