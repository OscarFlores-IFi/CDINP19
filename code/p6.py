# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 09:17:19 2019

@author: if715029
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as sc
import pandas as pd

#%%
data = pd.read_excel('../data/Datos_2015.xlsx',sheet_name='Atemajac')

#%%
data = data.iloc[:,2:7].dropna()

#%%
D1 = sc.squareform(sc.pdist(data.iloc[:,2:],'euclidean'))

#%%
data_norm = (data-data.mean(axis=0))/data.std(axis=0)

#%%
plt.subplot(1,2,1)
plt.scatter(data['CO'],data['PM10'])
plt.axis('square')
plt.subplot(1,2,2)
plt.scatter(data_norm['CO'],data_norm['PM10'])
plt.axis('square')
plt.show()










