# -*- coding: utf-8 -*-
"""
Created on Thu May 30 20:38:50 2019

@author: Oscar Flores
"""

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn import svm 
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split #hacer la separacion de los datos de prueba y los reales
from sklearn.metrics import (accuracy_score,precision_score,recall_score)
import pickle #te permite guardar datos o variables, en este caso guardar el modelo entrenado

#%% Importar los datos 
data=pd.read_csv('../Archivos/creditcard.csv')
X=data.iloc[:,1:30]
Y=data.iloc[:,30]
del data #elimina una variable

#%% Normalizar Amount 
X['Amount']=(X['Amount']-X['Amount'].mean())/X['Amount'].std()

#%% Seleccionar los datos de entrenamiento y prueba 
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=0)    #random_state es para que todos tengamos lo mismo, test_train pones la cantidad de datos que quieres probar (por lo general es 30%)
del X,Y

#%% Regresion logistica (revisar el polinomio optimo)
modelo_rl=linear_model.LogisticRegression()
grados= np.arange(1,4)  #depende de lo que tu compu permita
Accu=np.zeros(grados.shape)
Prec=np.zeros(grados.shape)
Reca=np.zeros(grados.shape)

for ngrado in grados:
    poly=PolynomialFeatures(ngrado)     #crea el polinomio
    Xa=poly.fit_transform(X_train)      #el polinomio que creaste con los datos que tienes
    modelo_rl.fit(Xa,Y_train)           #entrenas el modelo
    Yhat=modelo_rl.predict(Xa)          #estimas los datos con el modelo entrenadi
    Accu[ngrado-1]=accuracy_score(Y_train,Yhat)     #mide los ceros y unos comunes (emparejamiento simple)
    Prec[ngrado-1]=precision_score(Y_train,Yhat) 
    Reca[ngrado-1]=recall_score(Y_train,Yhat) 
    
#%% Graficar el codo 
plt.plot(grados,Accu)
plt.plot(grados,Prec)
plt.plot(grados,Reca)
plt.xlabel('Grado del polinomio')
plt.ylabel('% Aciertos')
plt.legend(('Accu','Prec','Reca'),loc='best')
plt.grid()
plt.show()

#%% Modelo definitivo 
ngrado=2
poly=PolynomialFeatures(ngrado)
Xa=poly.fit_transform(X_train)
modelo_rl.fit(Xa,Y_train)
Yhat=modelo_rl.predict(Xa)
print(accuracy_score(Y_train,Yhat))
print(precision_score(Y_train,Yhat))
print(recall_score(Y_train,Yhat))

#%% Probar con datos desconocidos 
Xa_test=poly.fit_transform(X_test)
Yhat_rl_test=modelo_rl.predict(Xa_test)
print(accuracy_score(Y_test,Yhat_rl_test))
print(precision_score(Y_test,Yhat_rl_test)
print(recall_score(Y_test,Yhat_rl_test))

#%% 
#Probar el modelo SVM
modelo_svm=svm.SVC(kernel='rbf')
modelo_svm.it(X_train,Y_train)
Yhat_sv_train=modelo_svm.predict(X_train)
print(accuracy_score(Y_train,Yhat_sv_train))
print(precision_score(Y_train,Yhat_sv_train)
print(recall_score(Y_train,Yhat_sv_train))

#%% Evaluar los datos de test con SVM 
Yhat_sv_test=modelo_svm.predict(X_test)
print(accuracy_score(Y_test,Yhat_sv_test))
print(precision_score(Y_test,Yhat_sv_test)
print(recall_score(Y_test,Yhat_sv_test)

#%% guardar los modelos 
pickle.dump(modelo_rl,open('modelo_rl.sav','wb'))
pickle.dump(modelo_svm,open('modelo_svm.sav','wb'))

modelo=pickle.load('modelo_rl.sav')