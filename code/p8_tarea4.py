# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 09:57:28 2019

@author: if715029
"""

import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as hi
import matplotlib.pyplot as plt

#%% Seleccionar datos
data = pd.read_excel('../data/Test de películas(1-16).xlsx')

csel = np.arange(6,243,3)
cnames = list(data.columns.values[csel])
datan = data[cnames]

#%% Dendograma
Z = hi.linkage(datan,metric='euclidean',method='complete')

#%% Grafica de dendograma
hi.dendrogram(Z)

#%% Grafica de codo
last = Z[:,2]
last = last[::-1]
plt.plot(np.arange(len(last))+1,last)

#%% Usuario 1: Oscar Flores
sim = hi.fcluster(Z,6,criterion='maxclust') #pertenezco al grupo 4
pel_sim = datan[sim==4]







