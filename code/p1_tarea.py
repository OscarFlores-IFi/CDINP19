# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 20:08:02 2019

@author: if715029
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#%% 2. 

accidents = pd.read_csv('../data/Accidents_2015.csv')

#%% a)

lat = accidents['Latitude'].values
lon = accidents['Longitude'].values
plt.scatter(lon,lat)
plt.show()
plt.hexbin(lon,lat,bins=10,cmap='inferno')
plt.show()

#%% b)

day = accidents.Day_of_Week
day = pd.value_counts(day)
print(day)

#%% 3.

casualties = pd.read_csv('../data/Casualties_2015.csv')

#%% a)

sex = casualties.Sex_of_Casualty
sex = pd.value_counts(sex)
print(sex)

#%% b)

age = casualties['Age_of_Casualty']
plt.hist(age,bins=15);
plt.show()
#%% 4.

vehicles = pd.read_csv('../DATA/Vehicles_2015.csv')

#%% a)

sex = pd.value_counts(vehicles['Sex_of_Driver'])
print(sex)

#%% b)

age = vehicles['Age_of_Driver']
plt.hist(age,bins = 20)
plt.show()

#%%




