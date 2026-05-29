# -*- coding: utf-8 -*-
"""
Created on Fri May 29 15:37:42 2026

@author: Enzo
"""

###############################################################################
# PROYECTO PARA ELABORAR UNA CARACTERIZACIÓN DE BECARIOS DEL PRONABEC
###############################################################################

###############################################################################
# El proyecto sigue un enfoque de líbrerias integradas
###############################################################################

# Se importan las librerias que serán usadas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Se carga el archivo en formato xlsx denominado 0_BGB_2013_2025, el cual
# contiene información de los becarios, y se almacena en un objeto dataframe

pronabec = pd.read_excel("0_BGB_2013_2025.xlsx", sheet_name="Sheet1", header=0)

# Se identifican caracteristicas estructurales del dataframe pronabec
