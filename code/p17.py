# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 09:51:21 2019

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

#%% Buscar el grado de polinomio optimo
modelo = linear_model.LogisticRegression(C=1)
grados = np.arange(1,20)
Scores = np.zeros((len(grados),3)) #Dentro de info estará: Accuracy, Precision, Recall
nW = np.zeros(len(grados)) #cant. de 

for grado in grados:
    poly = PolynomialFeatures(grado)
    Xa = poly.fit_transform(X)
    modelo.fit(Xa,Y)
    Yhat = modelo.predict(Xa)
    Scores[grado-1] = [accuracy_score(Y,Yhat),
                    precision_score(Y,Yhat),
                    recall_score(Y,Yhat)]
    nW[grado-1] = len(modelo.coef_[0])

plt.plot(Scores)

#%% Seleccionar parámetros para reducir ecuación. 
ngrado = 5

poly = PolynomialFeatures(ngrado)
Xa = poly.fit_transform(X)
modelo = linear_model.LogisticRegression(C=1)
modelo.fit(Xa,Y)
Yhat = modelo.predict(Xa)


W = modelo.coef_[0]
plt.bar(np.arange(len(W)),W)
plt.grid()
plt.show()

#%%
umbral = .5
indx = np.abs(W) > umbral

Xa_simplificada = Xa[:,indx]
modelo_opt = linear_model.LogisticRegression(C=1)
modelo_opt.fit(Xa_simplificada,Y)
Yhat_opt = modelo_opt.predict(Xa_simplificada)

print([accuracy_score(Y,Yhat),
        precision_score(Y,Yhat),
        recall_score(Y,Yhat)])

print([accuracy_score(Y,Yhat_opt),
        precision_score(Y,Yhat_opt),
        recall_score(Y,Yhat_opt)])












