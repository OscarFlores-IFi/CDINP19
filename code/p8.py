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











