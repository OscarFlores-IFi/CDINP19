#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 09:22:00 2019

@author: edzna
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
import scipy.optimize as opt #paquetería de métodos numéricos

#%% Generar datos de deudores y pagadores
X,Y = make_blobs(n_samples=100,centers=[[0,0],[5,5]],cluster_std=[2.2,1.8],n_features=2) #n_features numero de variables.  

plt.scatter(X[:,0],X[:,1],c=Y)
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

#%% Función Logística
def fun_log(V):
    return 1/(1+np.exp(-V))

#%% Regresión Logística
def reg_log(W, X, Y):  #la 'Y' no es necesaria, pero la paquetería utilizada la requiere
    V = np.matrix(X)*np.matrix(W).transpose()
    return np.array(fun_log(V))[:,0]

#%% Función de costos
def fun_cost(W, X, Y):
    Y_est = reg_log(W, X, Y)
    J = np.sum(-Y*np.log(Y_est)-(1-Y)*np.log(1-Y_est))/len(Y)
    return J

#%% Inicializar variables para optimización.
Xa = np.append(np.ones((len(Y),1)),X,axis=1)
m,n = np.shape(Xa)
W = np.zeros(n)

#%% Optimización 
res = opt.minimize(fun_cost, W, args=(Xa, Y))
W = res.x

#%% Simular modelo
Y_est = np.round(reg_log(W,Xa,Y))






