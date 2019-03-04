# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 10:19:24 2019

@author: if715029
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
import scipy.spatial.distance as sc

#%% Generar datos aleatorios para clustering. 
np.random.seed(4711)
a = np.random.multivariate_normal([10,0],[[3,0],[0,3]],size=100)
b = np.random.multivariate_normal([0,20],[[3,0],[0,3]],size=100)
c = np.random.multivariate_normal([3,7],[[3,0],[0,3]],size=100)

X = np.concatenate((a,b,c))

#%% Graficarlos
plt.scatter(X.T[0],X.T[1])
plt.xlabel('x1')
plt.ylabel('x2')
plt.axis('square')
plt.grid()
plt.show()

#%% Aplicar algoritmo clustering
Z = hierarchy.linkage(X,metric='euclidean',method='complete')

plt.figure(figsize=(25,15))
plt.title('Dendograma')
plt.xlabel('Indice')
plt.ylabel('Distancia')
hierarchy.dendrogram(Z)
plt.show()

#%%
Z = hierarchy.linkage(X,metric='euclidean',method='complete')

plt.figure(figsize=(25,15))
plt.title('Dendograma')
plt.xlabel('Indice')
plt.ylabel('Distancia')
hierarchy.dendrogram(Z,truncate_mode='level',p=3)
plt.show()

#%%
Z = hierarchy.linkage(X,metric='euclidean',method='single')

plt.figure(figsize=(25,15))
plt.title('Dendograma')
plt.xlabel('Indice')
plt.ylabel('Distancia')
hierarchy.dendrogram(Z,truncate_mode='lastp',p=3)
plt.show()

#%% Grafica de codo
last = Z[-15:,2]
last = last[::-1]

plt.plot(np.arange(len(last))+1,last)
# codo en 2, 4 o 10 grupos.

#%% Gradiente

grad = np.abs(np.diff(last))
plt.plot(np.arange(1,len(last))+1,grad)
# codo en 2 o 4 grupos. 

#%% Seleccionar elementos de los grupos formados
ngrupos = 4
grupos = hierarchy.fcluster(Z,ngrupos,criterion='maxclust')

plt.figure()
plt.scatter(X[:,0],X[:,1],c=grupos,cmap=plt.cm.tab20_r)

#%%
distance = 1
grupos = hierarchy.fcluster(Z,distance,criterion='distance')

plt.figure()
plt.scatter(X[:,0],X[:,1],c=grupos,cmap=plt.cm.prism)

#%%
idx = grupos == 5
subdata = X[idx,:]
















