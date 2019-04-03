#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mylib import mylib

#%% 
data = pd.read_csv('tpeunacional2015.csv',encoding='latin-1')

#%% dropna's
data = data.iloc[:,:-1]

#%% DQR
mireporte = mylib.dqr(data) #Reporte de calidad de los datos.

#%% Trabajadores por Municipio
municipios = pd.unique(data.Municipio) # lista de municipios disponibles. 
clas_trabajador = ['Trabajadores_eventuales_urbanos', 'Trabajadores_permanentes',
       'Trabajadores_eventuales_del_campo', 'Trabajadores_Asegurados'] #clasificacion de trabajadores (únicamente los que tienen datos 'int' si el tipo de dato es 'object' no funciona.)

Trabajadores_Municipio = np.zeros((len(municipios),len(clas_trabajador))) #matriz donde se guardarán todos los trabajadores de cada municipio. 
for j in np.arange(len(clas_trabajador)):
    T = np.zeros(len(municipios)) #vector vacío
    for i in np.arange(len(municipios)):
        T[i] = data[clas_trabajador[j]][data.Municipio == municipios[i]].sum() #se suman los tipos de trabajador 'j' en cada municipio 'i'.
    
    Trabajadores_Municipio[:,j] = T #Se guardan los datos en la matriz 
    
    T_M = pd.DataFrame(T,index=municipios) #DataFrame con municipios y cant. de trabajadores tipo 'j'
    T_M = T_M.sort_values(by=0,ascending=False) #Se ordenan los datos de mayor a menor.
    
    Fig = plt.figure() #Se crea una figura
    T_M.iloc[-21:-1,-1].plot(kind='bar');
    plt.title(clas_trabajador[j]) #Se le pone el título
    plt.show()  
    Fig.savefig('MU_'+clas_trabajador[j],bbox_inches='tight') #Se guarda con el nombre de la categoría.
    
    Fig = plt.figure() #Se crea una figura    
    T_M.iloc[0:20,0].plot(kind='bar'); #Se grafica en barras los primeros 20 municipios
    plt.title(clas_trabajador[j]) #Se le pone el título
    plt.show() 
    Fig.savefig('MP_'+clas_trabajador[j],bbox_inches='tight') #Se guarda con el nombre de la categoría. 

Trabajadores_Municipio = pd.DataFrame(Trabajadores_Municipio, index= municipios)

#%% Trabajadores por actividad
actividad = pd.unique(data.Division_de_Actividad)
clas_trabajador = ['Trabajadores_eventuales_urbanos', 'Trabajadores_permanentes',
       'Trabajadores_eventuales_del_campo', 'Trabajadores_Asegurados'] #clasificacion de trabajadores (únicamente los que tienen datos 'int' si el tipo de dato es 'object' no funciona.)

Trabajadores_Actividad = np.zeros((len(actividad),len(clas_trabajador))) #matriz donde se guardarán todos los trabajadores de cada municipio. 
for j in np.arange(len(clas_trabajador)):
    T = np.zeros(len(actividad)) #vector vacío
    for i in np.arange(len(actividad)):
        T[i] = data[clas_trabajador[j]][data.Division_de_Actividad == actividad[i]].sum()
    Trabajadores_Actividad[:,j] = T

    T_A = pd.DataFrame(T,index=actividad) #DataFrame con municipios y cant. de trabajadores tipo 'j'
    T_A = T_A.sort_values(by=0,ascending=False) #Se ordenan los datos de mayor a menor.
    
    Fig = plt.figure() #Se crea una figura
    T_A.iloc[0:20,0].plot(kind='bar'); #Se grafica en barras los primeros 20 municipios
    plt.title(clas_trabajador[j]) #Se le pone el título
    plt.show() 
    Fig.savefig('A_'+clas_trabajador[j],bbox_inches='tight') #Se guarda con el nombre de la categoría.
    
Trabajadores_Actividad = pd.DataFrame(Trabajadores_Actividad, index = actividad)

#%% Trabajadores por Estado
estado = pd.unique(data.Entidad_Federativa) 
clas_trabajador = ['Trabajadores_eventuales_urbanos', 'Trabajadores_permanentes',
       'Trabajadores_eventuales_del_campo', 'Trabajadores_Asegurados'] #clasificacion de trabajadores (únicamente los que tienen datos 'int' si el tipo de dato es 'object' no funciona.)

Trabajadores_Estado = np.zeros((len(estado),len(clas_trabajador))) #matriz donde se guardarán todos los trabajadores de cada municipio. 
for j in np.arange(len(clas_trabajador)):
    T = np.zeros(len(estado)) #vector vacío
    for i in np.arange(len(estado)):
        T[i] = data[clas_trabajador[j]][data.Entidad_Federativa == estado[i]].sum()
    Trabajadores_Estado[:,j] = T

    T_E = pd.DataFrame(T,index=estado) #DataFrame con municipios y cant. de trabajadores tipo 'j'
    T_E = T_E.sort_values(by=0,ascending=False) #Se ordenan los datos de mayor a menor.
    
    Fig = plt.figure() #Se crea una figura
    T_E.iloc[:,0].plot(kind='bar'); #Se grafica en barras los primeros 20 municipios
    plt.title(clas_trabajador[j]) #Se le pone el título
    plt.show() 
    Fig.savefig('E_'+clas_trabajador[j],bbox_inches='tight') #Se guarda con el nombre de la categoría.
    
Trabajadores_Estado = pd.DataFrame(Trabajadores_Estado, index = estado) #añadir index con estados

#%% Actividad porcentual
AP = Trabajadores_Actividad.iloc[:,3]/Trabajadores_Actividad.iloc[:,0:3].sum(axis=1)
AP = AP.sort_values(ascending=False) #Se ordenan los datos de mayor a menor.  

Fig = plt.figure() #Se crea una figura
AP.plot(kind='bar'); #Se grafica en barras los primeros 20 municipios
plt.title(clas_trabajador[j]) #Se le pone el título
plt.show() 
Fig.savefig('%Actividad_'+clas_trabajador[j]) #Se guarda con el nombre de la categoría.
    
#%% Estados porcentual
EP = Trabajadores_Estado.iloc[:,3]/Trabajadores_Estado.iloc[:,0:3].sum(axis=1)
EP = EP.sort_values(ascending=False) #Se ordenan los datos de mayor a menor. 
   
Fig = plt.figure() #Se crea una figura
EP.plot(kind='bar'); #Se grafica en barras los primeros 20 municipios
plt.title(clas_trabajador[j]) #Se le pone el título
plt.show() 
Fig.savefig('%Estados_'+clas_trabajador[j]) #Se guarda con el nombre de la categoría.
