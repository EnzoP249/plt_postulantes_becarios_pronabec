# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 11:16:56 2026

@author: Enzo
"""

###############################################################################
# PROYECTO PARA UN ESQUEMA DE MATCHING USANDO BECARIOS DEL PRONABEC
###############################################################################

###############################################################################
# OBJETIVO: ESTE PROYECTO TIENE COMO FINALIDAD REALIZAR UN MATCHING ENTRE
# PROGRAMA DE MAESTRIA Y DOCTORADO CON AREAS PRIORITARIAS DEFINIDAS POR EL
# CONCYTEC
###############################################################################


###############################################################################
# El proyecto sigue un enfoque de líbrerias integradas
###############################################################################

# Se importan las librerias que serán usadas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from rapidfuzz import process, fuzz
from unidecode import unidecode
import geopandas
import re
from unidecode import unidecode
from rapidfuzz import process, fuzz
from shapely.geometry import Point, Polygon
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mticker
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)


###############################################################################
# Se describen los colores que integran la paleta institucional para mis gráficos
###############################################################################

#1. Celeste claro
#HEX: #5FB7C6
#Nombre descriptivo: Celeste muy claro
#Uso: fondos, áreas suaves, mapas base

#2. Verde olivo
#HEX: #A3AD2C
#Nombre descriptivo: Verde olivo institucional
#Uso: color principal de datos (barras, líneas)

#3. Azul petróleo
#HEX: #0B4F6C
#Nombre descriptivo: Azul petróleo
#Uso: énfasis, títulos, bordes


# Se carga el archivo en formato xlsx denominado 0_BGB_2013_2025, un nuevo archivo enviado por el pronabec, el cual
# contiene información de los becarios. El archivo se almacena en un objeto dataframe

# Se construye una función que aborde la conversión de int en str para un procesamiento óptimizado
def int_to_str(value):
    return str(value)

# Especifica el diccionario de conversión en el parámetro converters
converters = {"ID_POSTULACION": int_to_str}

pronabec = pd.read_excel("BGB_2013_2025_VF_innominado.xlsx", sheet_name="BGB", header=0, converters=converters)

# Se identifican caracteristicas estructurales del dataframe pronabec
pronabec.shape
pronabec.columns
pronabec.info()
pronabec.dtypes
pronabec.head(10)
pronabec["CONDICION_FINAL"].head(10)

# Se realiza un análisis detallado de la columna ID_POSTULACION del dataframe pronabec
# Se identifica la presencia de nulos en la columna ID_POSTULACION del dataframe pronabec
nulo = pronabec["ID_POSTULACION"].isna().sum()
print(f"la columna ID_POSTULACION contiene {nulo} valores nulos")

# Se identifica los valores únicos de la columna ID_POSTULACION del dataframe pronabec
pronabec["ID_POSTULACION"].nunique()