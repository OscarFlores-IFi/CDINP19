# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 09:49:04 2019

@author: if715029
"""

import pandas as pd
import numpy as np

#%%
accidents = pd.read_csv('../data/Accidents_2015.csv')
columns = pd.DataFrame(list(accidents.columns.values),columns=['Nombres'],index=list(accidents.columns.values))
d_types = pd.DataFrame(accidents.dtypes,columns=['tipo'])
missing_val = pd.DataFrame(accidents.isnull().sum(),columns=['Valores perdidos'])
present_values = pd.DataFrame(accidents.count(),columns=['Valores presentes'])
unique_values = pd.DataFrame(accidents.nunique(),columns=['Valores unicos'])


min_values = pd.DataFrame(columns=['Min'])
max_values = pd.DataFrame(columns=['Max'])
for col in list(accidents.columns.values):
    try:
        min_values.loc[col] = [accidents[col].min()]
        max_values.loc[col] = [accidents[col].max()]
    except:
        pass




#%%


reporte_calidad_datos = columns.join(d_types).join(missing_val).join(present_values).join(unique_values).join(min_values).join(max_values)