#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 19:14:33 2019

@author: fh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as sc

#%% Importar datos. 
cancelacion = pd.read_csv('cancelacion_2017.csv',encoding='latin-1')
dirty1 = pd.read_csv('dirty_Info_Alumnos_v1.csv',encoding='latin-1')
dirty2 = pd.read_csv('dirty_Info_Alumnos_v2.csv',encoding='latin-1')
enfermedades = pd.read_csv('enfermedades.csv',index_col='nombre')

#%% Uso de suelo.
uso_suelo = pd.value_counts(cancelacion['Uso Suelo'])

plt.figure(figsize=(16,4))
uso_suelo.plot(kind='bar')
plt.show()

#%% Municipios con uso de suelo habitacional. 
municipios = cancelacion[cancelacion['Uso Suelo'] == 'HABITACIONAL']
municipios_ = pd.value_counts(municipios['Municipio'])

plt.figure(figsize=(16,3.5))
municipios_.plot(kind='bar')
plt.show()

#%% Dejar únicamente datos con dígitos en teléfono. 
clean2 = dirty2.copy()

clean2.iloc[:,-1] = clean2.iloc[:,-1].apply(only_digits) #para correr esta parte tienen que haber sido definidas las funciones previamente. 

#%% Eliminar números telefónicos con menos de 10 dígitos. 
for i in range(len(clean2)):
    if len(str(clean2.iloc[i,-1])) != 10:
        clean2.iloc[i,-1] = 'missing'
        
#%% Eliminar expedientes incorrectos. 
for i in range(len(clean2)):
    if len(str(clean2.iloc[i,3])) != 6:
        clean2.iloc[i,3] = 'missing'

#%% Limpiar semestre
clean2['sem estre'] = clean2['sem estre'].apply(only_digits)

#%% Limpiar nombres
clean2.iloc[:,0] = clean2.iloc[:,0].apply(replace,args=('¾','Ñ'))
esp = ['&','_','-','?','ð',':P',':0','x0',':)','%','/','.',':','     ','    ','   ','  ']
for i in esp: 
    clean2.iloc[:,0] = clean2.iloc[:,0].apply(replace,args=(i,' '))
clean2.iloc[:,0] = clean2.iloc[:,0].apply(replace,args=('0','O'))
clean2.iloc[:,0] = clean2.iloc[:,0].apply(replace,args=('PEDRQ','PEDRO'))
clean2.iloc[:,0] = clean2.iloc[:,0].apply(remove_punctuation)
clean2.iloc[:,0] = clean2.iloc[:,0].apply(remove_digits)

#%% Enfermedades

#%% Cambiar orden, de izquierda a derecha. 
enf = np.zeros((5,6))
for i in range(len(enf)):
    enf[i,:] = enfermedades.iloc[:,4-i]
enf = enf.T

#%% Graficar cambios porcentuales en las enfermedades. 
pct = enf[:,1:]/enf[:,:-1]-1
pct = pd.DataFrame(pct)
pct.T.plot()

#%% Cambios porcentuales más parecidos. 
norm = (pct-pct.mean(axis=0))/pct.std(axis=0)
dist = sc.squareform(sc.pdist(norm,'euclidean'))

#%%
enf = enf.T
enf1 = (enf-enf.mean(axis=0))/enf.std(axis=0)
anios = sc.squareform(sc.pdist(enf1,'euclidean'))




