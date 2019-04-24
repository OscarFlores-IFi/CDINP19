# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 09:33:04 2019

@author: if715029
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.metrics import (accuracy_score,precision_score,recall_score)

#%% Importar datos
data = pd.read_csv('../data/ex2data2.txt',header=None)

X = data.iloc[:,0:2]
Y = data.iloc[:,2]

plt.scatter(X[0],X[1],c=Y)
plt.show()

#%% Preparar datos (crear polinomio 'X*')
ngrado = 10 # grado del polinomio
poly = PolynomialFeatures(ngrado)
Xa = poly.fit_transform(X) # X*

#%% Crear y entrenar el modelo de clasificación
#modelo = linear_model.LogisticRegression(C=1e20) # Con sobre-ajuste.
modelo = linear_model.LogisticRegression(C=10) # Sin sobre-ajuste. 
modelo.fit(Xa,Y)

Yhat = modelo.predict(Xa)

#%% desempeño
accuracy_score(Y,Yhat)

#%% precisión
precision_score(Y,Yhat)

#%% recall
recall_score(Y,Yhat)

#%% ver modelo de prediccion
h = 0.01
x0min,x0max,x1min,x1max = X[0].min(),X[0].max(),X[1].min(),X[1].max()
xx,yy = np.meshgrid(np.arange(x0min,x0max,h),np.arange(x1min,x1max,h))

Xnew = pd.DataFrame(np.c_[xx.ravel(),yy.ravel()])

Xa_new = poly.fit_transform(Xnew)
Z = modelo.predict(Xa_new)
Z = Z.reshape(xx.shape)

plt.contour(xx,yy,Z)
plt.scatter(X[0],X[1],c=Y)
plt.show()

#%%
W = modelo.coef_
plt.bar(np.arange(len(W[0])),W[0])
plt.show()







