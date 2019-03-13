#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 10:12:41 2019

@author: Pepino
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd

#%% Importar los datos
data = pd.read_csv('../Data/creditcard.csv')