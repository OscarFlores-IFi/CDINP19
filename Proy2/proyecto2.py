# -*- coding: utf-8 -*-
"""
Created on Wed May  8 16:08:10 2019

@author: Oscar Flores
"""
from mylib import mylib
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn import svm 
#from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split 
from sklearn.metrics import (accuracy_score,precision_score,recall_score)
#import string
#import pickle 
#%%
data = pd.read_csv('Audit.csv')
data = data.drop('LOCATION_ID',axis=1)
data = data.drop('Detection_Risk',axis=1)
data = data.dropna()

mireporte = mylib.dqr(data) 


#%%############################################################################
#################################### MODELOS ##################################
###############################################################################
# Separar datos de entranamiento y prueba
X=data.iloc[:,:-1]
Y=data.iloc[:,-1]

X=(X-X.mean())/X.std()

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=0)    


#%%############################# E. MODELO RL #################################
modelo = linear_model.LogisticRegression()
grados = np.arange(1,5)
Accu = np.zeros(grados.shape)
Prec = np.zeros(grados.shape)
Rec = np.zeros(grados.shape)
nW = np.zeros(grados.shape)

for ngrado in grados:
    poly = PolynomialFeatures(ngrado)
    Xa_train = poly.fit_transform(X_train)
    modelo.fit(Xa_train,Y_train)
    Yhat_train = modelo.predict(Xa_train)
    Accu[ngrado-1] = accuracy_score(Y_train,Yhat_train)
    Prec[ngrado-1] = precision_score(Y_train,Yhat_train)
    Rec[ngrado-1] = recall_score(Y_train,Yhat_train)
    nW[ngrado-1] = len(modelo.coef_[0])
    

#%%     
plt.plot(grados,Accu)
plt.plot(grados,Prec)
plt.plot(grados,Rec)  
plt.title('Selección de Polinomio')
plt.xlabel('grado polinomio')
plt.ylabel('% aciertos')  
plt.legend(('Accuracy', 'Precision','Recall'),loc='best')
plt.grid()
plt.show()

#%%
ngrado = 2
poly = PolynomialFeatures(ngrado)
Xa_train = poly.fit_transform(X_train)
Xa_test = poly.fit_transform(X_test)
modelo = linear_model.LogisticRegression()
modelo = modelo.fit(Xa_train,Y_train)

#%%
Yhat_train = modelo.predict(Xa_train)

Accu = accuracy_score(Y_train,Yhat_train)
Prec = precision_score(Y_train,Yhat_train)
Rec = recall_score(Y_train,Yhat_train)

print([Accu,Prec,Rec])

#%%
Yhat_test = modelo.predict(Xa_test)

Accu = accuracy_score(Y_test,Yhat_test)
Prec = precision_score(Y_test,Yhat_test)
Rec = recall_score(Y_test,Yhat_test)

#%%
print([Accu,Prec,Rec])
plt.bar(['Accu','Prec','Rec'],[Accu,Prec,Rec])
plt.title('Regresión Logística')
plt.show()
#%%########################## E. MODELO SVM linear ############################
modelo = svm.SVC(kernel = 'linear')
modelo.fit(X_train,Y_train)
Yhat_train = modelo.predict(X_train)

Accu = accuracy_score(Y_train,Yhat_train)
Prec = precision_score(Y_train,Yhat_train)
Rec = recall_score(Y_train,Yhat_train)

print([Accu,Prec,Rec])

#%%
Yhat_test = modelo.predict(X_test)

Accu = accuracy_score(Y_test,Yhat_test)
Prec = precision_score(Y_test,Yhat_test)
Rec = recall_score(Y_test,Yhat_test)

print([Accu,Prec,Rec])
plt.bar(['Accu','Prec','Rec'],[Accu,Prec,Rec])
plt.title('SVM - Linear')
plt.show()

#%%########################### E. MODELO SVM poly #############################
modelo = svm.SVC(kernel = 'poly',degree=2)
modelo.fit(X_train,Y_train)
Yhat_train = modelo.predict(X_train)

Accu = accuracy_score(Y_train,Yhat_train)
Prec = precision_score(Y_train,Yhat_train)
Rec = recall_score(Y_train,Yhat_train)

print([Accu,Prec,Rec])

#%%
Yhat_test = modelo.predict(X_test)

Accu = accuracy_score(Y_test,Yhat_test)
Prec = precision_score(Y_test,Yhat_test)
Rec = recall_score(Y_test,Yhat_test)

print([Accu,Prec,Rec])
plt.bar(['Accu','Prec','Rec'],[Accu,Prec,Rec])
plt.title('SVM - Poly')
plt.show()


#%%########################### E. MODELO SVM rbf ##############################
modelo = svm.SVC(kernel = 'rbf')
modelo.fit(X_train,Y_train)
Yhat_train = modelo.predict(X_train)

Accu = accuracy_score(Y_train,Yhat_train)
Prec = precision_score(Y_train,Yhat_train)
Rec = recall_score(Y_train,Yhat_train)

print([Accu,Prec,Rec])

#%%
Yhat_test = modelo.predict(X_test)

Accu = accuracy_score(Y_test,Yhat_test)
Prec = precision_score(Y_test,Yhat_test)
Rec = recall_score(Y_test,Yhat_test)


print([Accu,Prec,Rec])
plt.bar(['Accu','Prec','Rec'],[Accu,Prec,Rec])
plt.title('SVM - Radial Basis Function')
plt.show()



#%%############################################################################
###################################### PCA ####################################
###############################################################################

X=data.iloc[:,:-1]
Y=data.iloc[:,-1]

media = np.mean(X)
X_m = X-media
X_cov = np.cov(X_m,rowvar=False) #matriz covarianzas

w,v = np.linalg.eig(X_cov) 


#%%
porcentaje = w/np.sum(w) # porcentaje de los datos que cuenta cada capa
porcentaje_acum = np.cumsum(porcentaje)

#%%
sel = 7

componentes = w[0:7]
M_trans = (v[:,0:7])

X_new = np.matrix(X_m)*np.matrix(M_trans)

#%%
X_new=(X_new-X_new.mean())/X_new.std()

X_new_train,X_new_test,Y_train,Y_test=train_test_split(X_new,Y,test_size=0.3,
                                                       random_state=0)    


#%%########################## E. MODELO RL (PCA)###############################
modelo = linear_model.LogisticRegression()
grados = np.arange(1,5)
Accu = np.zeros(grados.shape)
Prec = np.zeros(grados.shape)
Rec = np.zeros(grados.shape)
nW = np.zeros(grados.shape)

for ngrado in grados:
    poly = PolynomialFeatures(ngrado)
    X_newa_train = poly.fit_transform(X_new_train)
    modelo.fit(X_newa_train,Y_train)
    Yhat_train = modelo.predict(X_newa_train)
    Accu[ngrado-1] = accuracy_score(Y_train,Yhat_train)
    Prec[ngrado-1] = precision_score(Y_train,Yhat_train)
    Rec[ngrado-1] = recall_score(Y_train,Yhat_train)
    nW[ngrado-1] = len(modelo.coef_[0])
    

#%%     
plt.plot(grados,Accu)
plt.plot(grados,Prec)
plt.plot(grados,Rec)  
plt.title('Selección de Polinomio')
plt.xlabel('grado polinomio')
plt.ylabel('% aciertos')  
plt.legend(('Accuracy', 'Precision','Recall'),loc='best')
plt.grid()
plt.show()

#%%
ngrado = 2
poly = PolynomialFeatures(ngrado)
X_newa_train = poly.fit_transform(X_new_train)
X_newa_test = poly.fit_transform(X_new_test)
modelo = linear_model.LogisticRegression()
modelo = modelo.fit(X_newa_train,Y_train)

#%%
Yhat_train = modelo.predict(X_newa_train)

Accu = accuracy_score(Y_train,Yhat_train)
Prec = precision_score(Y_train,Yhat_train)
Rec = recall_score(Y_train,Yhat_train)

print([Accu,Prec,Rec])

#%%
Yhat_test = modelo.predict(X_newa_test)

Accu = accuracy_score(Y_test,Yhat_test)
Prec = precision_score(Y_test,Yhat_test)
Rec = recall_score(Y_test,Yhat_test)

#%%
print([Accu,Prec,Rec])
plt.bar(['Accu','Prec','Rec'],[Accu,Prec,Rec])
plt.title('Regresión Logística')
plt.show()
#%%####################### E. MODELO SVM linear (PCA)##########################
modelo = svm.SVC(kernel = 'linear')
modelo.fit(X_new_train,Y_train)
Yhat_train = modelo.predict(X_new_train)

Accu = accuracy_score(Y_train,Yhat_train)
Prec = precision_score(Y_train,Yhat_train)
Rec = recall_score(Y_train,Yhat_train)

print([Accu,Prec,Rec])

#%%
Yhat_test = modelo.predict(X_new_test)

Accu = accuracy_score(Y_test,Yhat_test)
Prec = precision_score(Y_test,Yhat_test)
Rec = recall_score(Y_test,Yhat_test)

print([Accu,Prec,Rec])
plt.bar(['Accu','Prec','Rec'],[Accu,Prec,Rec])
plt.title('SVM - Linear')
plt.show()

#%%########################### E. MODELO SVM poly (PCA)#############################
modelo = svm.SVC(kernel = 'poly',degree=2)
modelo.fit(X_new_train,Y_train)
Yhat_train = modelo.predict(X_new_train)

Accu = accuracy_score(Y_train,Yhat_train)
Prec = precision_score(Y_train,Yhat_train)
Rec = recall_score(Y_train,Yhat_train)

print([Accu,Prec,Rec])

#%%
Yhat_test = modelo.predict(X_new_test)

Accu = accuracy_score(Y_test,Yhat_test)
Prec = precision_score(Y_test,Yhat_test)
Rec = recall_score(Y_test,Yhat_test)

print([Accu,Prec,Rec])
plt.bar(['Accu','Prec','Rec'],[Accu,Prec,Rec])
plt.title('SVM - Poly')
plt.show()


#%%######################## E. MODELO SVM rbf (PCA)############################
modelo = svm.SVC(kernel = 'rbf')
modelo.fit(X_new_train,Y_train)
Yhat_train = modelo.predict(X_new_train)

Accu = accuracy_score(Y_train,Yhat_train)
Prec = precision_score(Y_train,Yhat_train)
Rec = recall_score(Y_train,Yhat_train)

print([Accu,Prec,Rec])

#%%
Yhat_test = modelo.predict(X_new_test)

Accu = accuracy_score(Y_test,Yhat_test)
Prec = precision_score(Y_test,Yhat_test)
Rec = recall_score(Y_test,Yhat_test)


print([Accu,Prec,Rec])
plt.bar(['Accu','Prec','Rec'],[Accu,Prec,Rec])
plt.title('SVM - Radial Basis Function')
plt.show()



#%%
X_recuperada = np.matrix(X_new)*np.matrix(M_trans.transpose())
X_recuperada = X_recuperada+np.ones(X.shape)*media.values




#%%############################################################################
##################################### TEST ####################################
###############################################################################

# Separar datos de enranamiento y prueba
X=data.iloc[:,:-1]
Y=data.iloc[:,-1]


TEST = pd.read_csv('Test_points.csv')
TEST = TEST.drop('LOCATION_ID',axis=1)
TEST = TEST.drop('Detection_Risk',axis=1)
TEST = TEST.dropna()
#%%############################# E. MODELO RL #################################
ngrado = 2
poly = PolynomialFeatures(ngrado)
Xa = poly.fit_transform(X)
Xa_test = poly.fit_transform(TEST)
modelo = linear_model.LogisticRegression()
modelo = modelo.fit(Xa,Y)

#%%
Yhat = modelo.predict(Xa)

Accu = accuracy_score(Y,Yhat)
Prec = precision_score(Y,Yhat)
Rec = recall_score(Y,Yhat)

#%%
Yhat_test_rl = modelo.predict(Xa_test)

#%%
print([Accu,Prec,Rec])
plt.bar(['Accu','Prec','Rec'],[Accu,Prec,Rec])
plt.title('Regresión Logística')
plt.show()
print(Yhat_test_rl)
#%%########################## E. MODELO SVM linear ############################
modelo = svm.SVC(kernel = 'linear')
modelo.fit(X,Y)
Yhat = modelo.predict(X)

Accu = accuracy_score(Y,Yhat)
Prec = precision_score(Y,Yhat)
Rec = recall_score(Y,Yhat)

print([Accu,Prec,Rec])

#%%
Yhat_test_svm = modelo.predict(TEST)


print([Accu,Prec,Rec])
plt.bar(['Accu','Prec','Rec'],[Accu,Prec,Rec])
plt.title('SVM - Linear')
plt.show()
print(Yhat_test_svm)

#%%############################## comprobación ################################

print(Yhat_test_rl+Yhat_test_svm)


