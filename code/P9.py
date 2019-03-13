#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 09:19:31 2019

@author: Pepino
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

#%% Generar los datos para clustering
semilla = 1500
X,Y = make_blobs(n_samples=1000,random_state=semilla)

plt.scatter(X[:,0],X[:,1])
plt.show()

#%% Aplicar KMeans
model = KMeans(n_clusters=3,random_state=semilla,init='random') #Se configura el algoritmo

modelo = model.fit(X) # Aqui se ejecuta e algoritmo

Ypredict = model.predict(X) #Se utiliza el algoritmo
centroides = model.cluster_centers_  #Esto es para ver donde esta el centroide
J = model.inertia_

# Visualizar los datos
plt.scatter(X[:,0],X[:,1],c=Ypredict)
plt.plot(centroides[:,0],centroides[:,1],'x')
plt.show()


#%% Criterios de desicion del numero de grupos
# Grafica de codo

inercias = np.zeros(10)
for k in np.arange(len(inercias))+1:
    model = KMeans(n_clusters=k,random_state=semilla,init='random')
    modelo = model.fit(X)
    inercias[k-1] = model.inertia_
    
plt.plot(np.arange(len(inercias))+1,inercias)
plt.xlabel('Num de grupos')
plt.ylabel('Inercia total')
plt.show()
