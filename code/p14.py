# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 09:32:09 2019

@author: if715029
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import sklearn.metrics as skm
import scipy.spatial.distance as sc 

#%%

digits = datasets.load_digits()

#%%
ndig = 100
for k in np.arange(ndig):
    plt.subplot(np.floor(np.sqrt(ndig)),np.ceil(np.sqrt(ndig)),k+1)
    plt.axis('off')
    plt.imshow(digits.images[k],cmap=plt.cm.gray_r)
    
#%% PCA en la base de datos
data = digits.data
media = data.mean(axis=0)    
data_m = data-media
M_cov = np.cov(data_m,rowvar=False)    
w,v = np.linalg.eig(M_cov)

#%% Decidir numero de variables a reducir
porcentaje = w/np.sum(w)
porcentaje_acum = np.cumsum(porcentaje)

limite = .50

plt.figure(figsize=(6.3,7))
plt.bar(np.arange(len(porcentaje_acum)),porcentaje_acum)
plt.bar(np.arange(len(porcentaje)),porcentaje)
plt.hlines(limite,0,64,'r')
plt.show()

#%% Proyectar datos en nuevas dimensiones
indx = porcentaje_acum<=limite
componentes = w[indx]
M_trans = v[:,indx]

data_new = np.array(np.matrix(data_m)*np.matrix(M_trans))

#%% Recuperar imagenes de las variables reducidas
data_r = np.array(np.matrix(data_new)*np.matrix(M_trans.transpose()))
data_r = data_r + media
data_r[data_r<0] = 0


#%%
ndig = 100
for k in np.arange(ndig):
    plt.subplot(np.floor(np.sqrt(ndig)),np.ceil(np.sqrt(ndig)),k+1)
    plt.axis('off')
    plt.imshow(np.reshape(data_r[k,:],(8,8)),cmap=plt.cm.gray_r)
    
    


#%% si decidimos hacerlo con 2 variables
M_trans = v[:,0:2]
data_new = np.array(np.matrix(data_m)*np.matrix(M_trans))
plt.scatter(data_new[:,0],data_new[:,1],c=digits.target)
plt.colorbar()
plt.grid()
plt.show()


#%% Reducir variables por metodo de varianza
varianza = np.var(data,axis=0)
plt.bar(np.arange(len(varianza)),varianza)
plt.show()

#%% Seleccionar nivel de varianza
nivel_varianza = 5
idnx = varianza > nivel_varianza

data_new = data[:,idnx]

img_f = np.zeros(64)
img_f[idnx] = 1
img_f = np.reshape(img_f,(8,8))

# Muestra los pixeles que varían más en la base de datos. 
plt.imshow(img_f,cmap=plt.cm.gray_r)
plt.show()






