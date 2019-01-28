# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 09:58:16 2019

@author: if715029
"""

from cdin import cdinp19
import pandas as pd

#%% Primera opción
accidents = pd.read_csv('../data/Accidents_2015.csv')
vehicles = pd.read_csv('../data/Vehicles_2015.csv')
casualties = pd.read_csv('../data/Casualties_2015.csv')
dqr_a = cdinp19.dqr(accidents)
dqr_v = cdinp19.dqr(vehicles)
dqr_c = cdinp19.dqr(casualties)

#%% Segunda opción
dqr = cdinp19.dqr

accidents = pd.read_csv('../data/Accidents_2015.csv')
vehicles = pd.read_csv('../data/Vehicles_2015.csv')
casualties = pd.read_csv('../data/Casualties_2015.csv')
dqr_a = dqr(accidents)
dqr_v = dqr(vehicles)
dqr_c = dqr(casualties)