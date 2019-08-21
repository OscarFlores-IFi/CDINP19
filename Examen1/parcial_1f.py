#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 20:57:29 2019

@author: fh
"""

#funciones utilizadas en parcial_1.py
import string
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
