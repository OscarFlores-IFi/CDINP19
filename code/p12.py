# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal
"""

import numpy as np
import matplotlib.pyplot as plt

#%% Creamos datos

datos = 10
data = np.array([np.arange(datos),np.arange(datos)]) - np.random.random((2,datos))*5
data = data.T

plt.scatter(data[:,0],data[:,1])

#%% Covertir datos a conjunto con media 0
media = data.mean(axis=0)
data_m = data-media

plt.scatter(data_m[:,0],data_m[:,1])
#%% Matriz de covarianzas
data_cov = np.cov(data_m,rowvar=False)

#%% Eigenvalues & Eigenvectors
# w = valores ; v = vectores
w,v = np.linalg.eig(data_cov)

#%% Dibujar vectores propios
x = np.arange(-5,5)
plt.scatter(data_m[:,0],data_m[:,1])
plt.plot(x, (v[1,0]/v[0,0])*x,'b--')
plt.plot(x, (v[1,1]/v[0,1])*x,'r--')
plt.axis('square')

#%% Transformar los ejes
M_trans = v[:,[1,0]]
componentes = w[[1,0]]

data_new = np.array(np.matrix(data_m)*np.matrix(M_trans))

plt.scatter(data_new[:,0],data_new[:,1])










