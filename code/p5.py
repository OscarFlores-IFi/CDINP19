# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 09:55:38 2019

@author: if715029
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as sc
#import pandas as pd

#%%
X = np.array([[2,3],[20,30],[-2,-3],[2,-3]])

plt.scatter(X[:,0],X[:,1])
plt.grid()
plt.show()

#%% Distancia euclideana
D1 = 1-sc.squareform(sc.pdist(X,'euclidean'))

#%% Indice coseno
D2 = 1-sc.squareform(sc.pdist(X,'cosine'))

#%%
D3 = 1-sc.squareform(sc.pdist(X,'correlation'))










