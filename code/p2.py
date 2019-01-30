# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 10:24:51 2019

@author: if715029
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import sklearn.metrics as skm

#%%

digits = datasets.load_digits()

#%%
ndig = 100
for k in np.arange(ndig):
    plt.subplot(np.floor(np.sqrt(ndig)),np.ceil(np.sqrt(ndig)),k+1)
    plt.axis('off')
    plt.imshow(digits.images[k],cmap=plt.cm.gray_r)





