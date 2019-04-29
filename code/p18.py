# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

#%%
np.random.seed(5)
X = np.r_[np.random.randn(20,2)-[2,2],np.random.randn(20,2)+[2,2]]
Y = [0]*20+[1]*20

plt.scatter(X[:,0],X[:,1],c=Y)
plt.show()

#%% Modelo de clasificación.
modelo = svm.SVC(kernel= 'linear')
#modelo = svm.SVC(kernel= 'poly', degree=2)
#modelo = svm.SVC(kernel= 'rbf')

modelo.fit(X,Y)

Yhat = modelo.predict(X)

#%% Dibujar vector soporte (aplica únicamente con modelo lineal, con polinomial o gausssiana no permite ver los polinomios)
W = modelo.coef_[0]
m = -W[0]/W[1]
xx = np.linspace(-4,4)
yy = m*xx-(modelo.intercept_[0]/W[1])

VS = modelo.support_vectors_

plt.plot(xx,yy, 'k--')
plt.scatter(X[:,0],X[:,1],c=Y)
plt.scatter(VS[:,0],VS[:,1],s=80,facecolors='k')
plt.show()


