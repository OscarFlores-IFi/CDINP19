# -*- coding: utf-8 -*-
"""
Created on Thu May 30 20:39:01 2019

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

#%% Aplicar clustering para encotnrar perfiles
inercias = np.zeros(10)
for k in np.arange(len(inercias)):
    model = KMeans(n_clusters=k+1,init='k-means++',random_state=0)
    model = model.fit(X_train)
    inercias[k] = model.inertia_

# Grafica de codo
plt.plot(np.arange(1,len(inercias)+1),inercias)
# Se decide tomar 4 grupos

#%%

model = KMeans(n_clusters=4,init='k-means++',random_state=0)
model = model.fit(X_train)
grupos = model.predict(X_train)
#%% Verificar si todos tienen 0's y 1's
for i in np.arange(model.n_clusters):
    print('grupo %s' %i)
    print(np.unique(Y_train[grupos==i])) #Vemos si todos los grupos tienen como resultado 1's y 0's.
    
#%% Definicion de modelo logístico
def graf_reglog(X,Y,grados): 
    modelo_rl=linear_model.LogisticRegression()
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

    plt.plot(grados,Accu)
    plt.plot(grados,Prec)
    plt.plot(grados,Reca)
    plt.xlabel('Grado del polinomio')
    plt.ylabel('% Aciertos')
    plt.legend(('Accu','Prec','Reca'),loc='best')
    plt.grid()
    plt.show()

    return(grados,Accu,Prec,Reca)
    
#%% Aplicar la función de reg_log
graf_reglog(X_train[grupos==0],Y_train[grupos==0],np.arange(1,4))
graf_reglog(X_train[grupos==1],Y_train[grupos==1],np.arange(1,4))
graf_reglog(X_train[grupos==2],Y_train[grupos==2],np.arange(1,4))
graf_reglog(X_train[grupos==3],Y_train[grupos==3],np.arange(1,4))

#%% Creando el grupo de modelos definitivo
Xa0 = PolynomialFeatures(2).fit_transform(X_train[grupos==0])
Xa1 = PolynomialFeatures(2).fit_transform(X_train[grupos==1])
Xa2 = PolynomialFeatures(2).fit_transform(X_train[grupos==2])
Xa3 = PolynomialFeatures(1).fit_transform(X_train[grupos==3])

modelo0 = linear_model.LogisticRegression().fit(Xa0,Y_train[grupos==0])
modelo1 = linear_model.LogisticRegression().fit(Xa1,Y_train[grupos==1])
modelo2 = linear_model.LogisticRegression().fit(Xa2,Y_train[grupos==2])
modelo3 = linear_model.LogisticRegression().fit(Xa3,Y_train[grupos==3])

#%% Evaluar el grupo de modelos
Yhat0 = modelo0.predict(Xa0)
Yhat1 = modelo1.predict(Xa1)
Yhat2 = modelo2.predict(Xa2)
Yhat3 = modelo3.predict(Xa3)
Yhat_total = np.zeros(Y_train.shape)

Yhat_total[grupos==0] = Yhat0
Yhat_total[grupos==1] = Yhat1
Yhat_total[grupos==2] = Yhat2
Yhat_total[grupos==3] = Yhat3

print(accuracy_score(Y_train,Yhat_total))
print(precision_score(Y_train,Yhat_total))
print(recall_score(Y_train,Yhat_total))




#%% Lo mismo pero con DATOS NUEVOS
grupos_test = model.predict(X_test)
Xa0 = PolynomialFeatures(2).fit_transform(X_test[grupos_test==0])
Xa1 = PolynomialFeatures(2).fit_transform(X_test[grupos_test==1])
Xa2 = PolynomialFeatures(2).fit_transform(X_test[grupos_test==2])
Xa3 = PolynomialFeatures(1).fit_transform(X_test[grupos_test==3])

Yhat0 = modelo0.predict(Xa0)
Yhat1 = modelo1.predict(Xa1)
Yhat2 = modelo2.predict(Xa2)
Yhat3 = modelo3.predict(Xa3)
Yhat_total_test = np.zeros(Y_test.shape)
Yhat_total_test[grupos_test==0] = Yhat0
Yhat_total_test[grupos_test==1] = Yhat1
Yhat_total_test[grupos_test==2] = Yhat2
Yhat_total_test[grupos_test==3] = Yhat3

print(accuracy_score(Y_test,Yhat_total_test))
print(precision_score(Y_test,Yhat_total_test))
print(recall_score(Y_test,Yhat_total_test))