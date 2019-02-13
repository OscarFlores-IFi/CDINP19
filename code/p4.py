# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 09:28:26 2019

@author: if715029
"""

import pandas as pd
import numpy as np
import scipy.spatial.distance as sc
from cdin import cdinp19 as mylib

#%% Importar datos
accidents = pd.read_csv('../data/Accidents_2015.csv')

#%% Data Quality Report
mireporte = mylib.dqr(accidents)

#%% Seleccionar columnas con datos enteros
indx = np.array(accidents.dtypes == 'int64')
col_list = list(accidents.columns.values[indx])
accidents_int = accidents[col_list]

mireporte = mylib.dqr(accidents_int)
#%% Seleccionar columnas con 10 o menos valores únicos. 
indx = np.array(mireporte['Valores unicos']<=10)
col_list_2 = np.array(col_list)[indx]
accidents_int_unique = accidents_int[col_list_2]

mireporte = mylib.dqr(accidents_int_unique)

#%% Obtener variables Dummies
accidents_dummy = pd.get_dummies(accidents_int_unique[col_list_2[0]],prefix=col_list_2[0])
for i in col_list_2[1:]:
    tmp = pd.get_dummies(accidents_int_unique[i],prefix=i)
    accidents_dummy = accidents_dummy.join(tmp)
#    print(pd.get_dummies(accidents_int_unique[i], prefix=i))



#%% Aplicar indices de similitud
D1 = sc.squareform(sc.pdist(accidents_dummy.iloc[0:30,:],'hamming'))
    












