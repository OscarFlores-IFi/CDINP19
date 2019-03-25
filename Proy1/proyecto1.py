#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 19:03:48 2019

@author: fh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 11:32:07 2019

@author: josean
"""

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

#%% Trabajadores por municipio
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
    T_M.iloc[0:20,0].plot(kind='bar'); #Se grafica en barras los primeros 20 municipios
    plt.title(clas_trabajador[j]) #Se le pone el título
    plt.show() 
    Fig.savefig(clas_trabajador[j]) #Se guarda con el nombre de la categoría. 
#%% grafica
    
#%% Trabajadores asegurados por municipio
#Trab_aseg = np.zeros(len(municipios))
#for i in np.arange(len(municipios)):
#    Trab_aseg[i] = data.Trabajadores_Asegurados[data.Municipio == municipios[i]].sum()
#
#T_a_m = pd.DataFrame(Trab_aseg,index=municipios)

#%% Trabajadores asegurados por actividad
#actividad = pd.unique(data.Division_de_Actividad)
#Trab_aseg_act = np.zeros(len(actividad))
#for i in np.arange(len(actividad)):
#    Trab_aseg_act[i] = data.Trabajadores_Asegurados[data.Division_de_Actividad == actividad[i]].sum()
#
#T_a_A = pd.DataFrame(Trab_aseg_act,index=actividad)




