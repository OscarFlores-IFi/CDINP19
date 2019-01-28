# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 09:22:55 2019

@author: if715029
"""
import pandas as pd
import numpy as np

def dqr(data):
    import pandas as pd
    import numpy as np
    
    
    columns = pd.DataFrame(list(data.columns.values),columns=['Nombres'],index=list(data.columns.values))
    d_types = pd.DataFrame(data.dtypes,columns=['tipo'])
    missing_val = pd.DataFrame(data.isnull().sum(),columns=['Valores perdidos'])
    present_values = pd.DataFrame(data.count(),columns=['Valores presentes'])
    unique_values = pd.DataFrame(data.nunique(),columns=['Valores unicos'])
    
    
    min_values = pd.DataFrame(columns=['Min'])
    max_values = pd.DataFrame(columns=['Max'])
    for col in list(data.columns.values):
        try:
            min_values.loc[col] = [data[col].min()]
            max_values.loc[col] = [data[col].max()]
        except:
            pass
    
    reporte_calidad_datos = columns.join(d_types).join(missing_val).join(present_values).join(unique_values).join(min_values).join(max_values)
    
    return (reporte_calidad_datos)
#%%


accidents = pd.read_csv('../data/Accidents_2015.csv')
vehicles = pd.read_csv('../data/Vehicles_2015.csv')
casualties = pd.read_csv('../data/Casualties_2015.csv')
dqr_a = dqr(accidents)
dqr_v = dqr(vehicles)
dqr_c = dqr(casualties)

#%%
Num_by_date = pd.DataFrame(pd.value_counts(accidents['Date']))
vehicles_day = pd.DataFrame(accidents.groupby(['Date'])['Number_of_Vehicles'].sum())
num_vehicles = pd.DataFrame(pd.value_counts(accidents['Number_of_Vehicles']))
casualties_day  = pd.DataFrame(accidents.groupby(['Date'])['Number_of_Casualties'].sum())

result = Num_by_date.join(vehicles_day).join(casualties_day)

vehicles_by_time = accidents.groupby(['Time'])['Number_of_Vehicles'].sum()
accidents_by_time = pd.DataFrame(pd.value_counts(accidents['Time']))

#%%
import matplotlib.pyplot as plt

plt.hist(accidents['Day_of_Week'],bins=7,normed=True)
plt.show()
