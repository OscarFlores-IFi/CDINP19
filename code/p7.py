# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 09:37:23 2019

@author: if715029
"""

# limpieza de base de datos y manejod de texto
import pandas as pd
import string
from datetime import datetime

#%% Importar tabla
dirty = pd.read_csv('../data//dirty_data_v3.csv', encoding='latin-1')

#%% Funcion para retirar signos de puntuación. 
def remove_punctuation(x):
    try:
        x = ''.join(ch for ch in x if ch not in string.puntuation)
    except:
        pass

    return(x)

#%% Remover digitos
def remove_digits(x):
    try:
        x = ''.join(ch for ch in x if ch not in string.digits)
    except:
        pass
    
    return(x)

#%% quitar espacios
def remove_whitespace(x):
    try:
        x = ''.join(x.split())
    except:
        pass
    
    return(x)
    
#%% reemplazar texto
def replace(x,to_replace,replacement):
    try:
        x = x.replace(to_replace,replacement)
    except:
        pass
    
    return(x)
    
#%% convertir a mayusculas
def uppercase_text(x):
    try:
        x = x.upper() 
    except:
        pass
    
    return (x)

#%%
def lowercase_text(x):
    try:
        x = x.lower()
    except:
        pass
    
    return(x)
    
#%%
def only_digits(x):
    try:
        x = ''.join(ch for ch in x if ch in string.digits)
    except:
        pass
    
    return(x)

#%% aplicar funciones
dirty['apellido'] = dirty['apellido'].apply(lowercase_text)
dirty['apellido'] = dirty['apellido'].apply(replace,args=('0','o'))
dirty['apellido'] = dirty['apellido'].apply(replace,args=('2','z'))
dirty['apellido'] = dirty['apellido'].apply(replace,args=('4','a'))
dirty['apellido'] = dirty['apellido'].apply(replace,args=('1','i'))
dirty['apellido'] = dirty['apellido'].apply(replace,args=('8','b'))
dirty['apellido'] = dirty['apellido'].apply(remove_punctuation)
dirty['apellido'] = dirty['apellido'].apply(remove_digits)











