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
import unicodedata
import pandas as pd


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

# Se eliminan los registros relacionados con el Perú dada la naturaleza deL esquema BGB
pronabec = pronabec[pronabec["PAISDESTINO"]!="PERU"]
# Se eliminan los registros que no cuentan con le pais de destino registrado
pronabec = pronabec[pronabec["PAISDESTINO"] != "SIN REGISTRO"]

# Se eliminan los registros cuyo nombre del programa es GENERICA
pronabec = pronabec[pronabec["NOMBRECARRERA"]!="GENERICA"]

# Se eliminan los registros cuyo nombre del programa es "Maestria" sin especficar de qué tipo
pronabec = pronabec[pronabec["NOMBRECARRERA"]!="MAESTRIA"]

# Se eliminan los registros cuyo nombre del programa es "Doctorado" sin especficar de qué tipo
pronabec = pronabec[pronabec["NOMBRECARRERA"]!="DOCTORADO"]

# Se eliminan registros menores a 18 años de edad
pronabec = pronabec[pronabec["EDADBASES"] >= 18]

# Se eliminan registros que en su estructura tienen la palabra bicentenario
pronabec = pronabec[~pronabec["NOMBRECARRERA"].str.contains("bicentenario", case=False, na=False)]
pronabec.columns

# Se construye un subset utilizando el dataframe pronabec
pronabec = pronabec[["AÑO_CONVOCATORIA", "ID_POSTULACION", "CONDICION_FINAL", "SEXO",
                     "EDADBASES", "PAISDESTINO", "NIVEL_EDUCATIVO", "INSTITUCION",
                     "TIPO_GESTION", "NOMBRECARRERA"]]


# Se renombra el atributo NOBRECARRERA por carrera_pronabec
pronabec.rename(columns=({"NOMBRECARRERA":"CARRERA_PRONABEC"}), inplace=True)

###############################################################################
# Se diseña e implementa un algoritmo de Clasificación Determinista basado en Léxico y Reglas jerarquicas
###############################################################################

# Se establece un conjunto de areas prioritarias bajo un enfoque de juicio de expertos
area_priorizada = ["Biotecnología", "biodiversidad"]

# Función para normalizar texto (remueve tildes y pasa a mayúsculas)
def normalizar_texto(texto):
    if not isinstance(texto, str):
        return ""
    # Descompone caracteres con tilde
    texto = unicodedata.normalize("NFD", texto)
    # Filtra caracteres de acento
    texto = "".join([c for c in texto if unicodedata.category(c) != "Mn"])
    return texto.upper().strip()


# Diccionario de reglas parsimoniosas (Raíces / Keywords por Área Priorizada)
# Permite capturar variaciones en español e inglés (ej. BIOTECH, BIODIVERSIDAD)
REGLAS_MATCHING = {
    "Biotecnología": [
        r"\bBIOTECNO",  # Biotecnología, Biotecnológica
        r"\bBIOTECH",  # Master of Biotechnology

    ],
    
    "Biodiversidad": [
        r"\bBIODIVERSI",  # Biodiversidad
        r"\bFAUNA\b",
        r"\bFLORA\b",
        r"\bECOLOGIA\b",
        r"\bCONSERVACION DE LA NATURALEZA\b",
        r"\bBIODIVERSIT",
        r"\bBIOLOGISCHE VIELFALT\b"
    ],
}


# Función de categorización fila por fila
def categorizar_carrera(carrera):
    carrera_norm = normalizar_texto(carrera)

    # Evalúa regla por regla en orden jerárquico
    for area, patrones in REGLAS_MATCHING.items():
        for patron in patrones:
            if pd.Series(carrera_norm).str.contains(patron, regex=True).iloc[0]:
                return area

    return "Otros"  # O None / "Otros"


# 4. Aplicación a la columna de tu DataFrame
pronabec['AREA_PRIORIZADA'] = pronabec['CARRERA_PRONABEC'].apply(categorizar_carrera)

pronabec["AREA_PRIORIZADA_BIN"] = pronabec["AREA_PRIORIZADA"].apply(
    lambda x: "NO" if x == "Otros" else "SI"
)


###############################################################################
# Se cosntruye un dataframe que almacena a los adjudicados
###############################################################################
adjudicado = pronabec[pronabec["CONDICION_FINAL"]=="SE LE ADJUDICÓ LA BECA"]

###############################################################################
# Se obtiene un dataframe que almacena a los becarios de programas de maestria
###############################################################################
becario_maestria = adjudicado[adjudicado["NIVEL_EDUCATIVO"]=="MAESTRIA"]
becario_maestria.columns

# Se analiza la participación de los programas de maestria asociadas con las areas priorizadas
becario_maestria.AREA_PRIORIZADA_BIN.value_counts(normalize=True).round(3)

###############################################################################
# Se analiza la evolución de las areas prioritarias con respecto al total
###############################################################################

# Configuración de estilo académico
sns.set_style("white")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"

# 1. Ajustar el nombre exacto de la columna de año (ej. 'CONVOCATORIA')
# Asumiendo 'CONVOCATORIA' o la primera columna observada en tu dataframe:
col_anio = "AÑO_CONVOCATORIA"  
col_target = "AREA_PRIORIZADA_BIN"  # O la columna binaria ('SI' / 'NO')

# 2. Calcular porcentajes por año (Tabla cruzada al 100%)
df_cross = (
    pd.crosstab(
        becario_maestria[col_anio], becario_maestria[col_target], normalize="index"
    )
    * 100
)

fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

# Colores institucionales con mayor contraste
color_si = "#E65100"  # Naranja sobrio (Área Priorizada)
color_no = "#B0BEC5"  # Gris neutro para la categoría de control / "NO"

# Invertir orden en la matriz para que 'SI' quede abajo en la base
cols_ordered = ["SI", "NO"]
df_plot = df_cross[cols_ordered]

# Coordenadas
x_labels = df_plot.index.astype(str)
x_pos = np.arange(len(x_labels))
width = 0.55

# Trazar barras apiladas
bar_si = ax.bar(
    x_pos,
    df_plot["SI"],
    width=width,
    color=color_si,
    label="Área Priorizada (SI)",
    edgecolor="white",
)
bar_no = ax.bar(
    x_pos,
    df_plot["NO"],
    bottom=df_plot["SI"],
    width=width,
    color=color_no,
    label="Otros (NO)",
    edgecolor="white",
)

# Anotación inteligente de valores para la categoría 'SI'
for i, val in enumerate(df_plot["SI"]):
    # Si el valor es pequeño, la etiqueta va arriba de la barra naranja
    if val < 4.0:
        ax.text(
            i,
            val + 1.2,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            color=color_si,
            fontsize=8.5,
            fontweight="bold",
        )
    else:
        ax.text(
            i,
            val / 2,
            f"{val:.1f}%",
            ha="center",
            va="center",
            color="white",
            fontsize=8.5,
            fontweight="bold",
        )

# Estética y límites
ax.set_ylim(0, 105)  # Espacio para etiquetas superiores
ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels, fontsize=9.5)
ax.set_ylabel("Porcentaje (%)", fontsize=10, labelpad=10)
ax.set_xlabel("Año de Convocatoria", fontsize=10, labelpad=10)

ax.set_title(
    "Evolución de la Participación en Áreas Priorizadas (2013-2025)",
    fontsize=12,
    fontweight="bold",
    pad=15,
)

# Grilla limpia
ax.yaxis.grid(True, linestyle="--", alpha=0.3, color="#CCCCCC")
ax.set_axisbelow(True)
sns.despine(top=True, right=True)

# Leyenda
ax.legend(
    title="Área Priorizada",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    frameon=True,
    facecolor="#F8F9FA",
    edgecolor="none",
    fontsize=9,
    title_fontsize=9,
)

plt.tight_layout()
plt.show()


# Este gráfico muestra la tasa de participación en "áreas priorizadas" desagregada por rango de edad y género

# 1. Crear la variable de rango etario según la lógica OCDE corregida
bins = [19, 29, 39, 120]
labels = ['20 - 29 años', '30 - 39 años', '40 - 50 años']

becario_maestria["RANGO_EDAD"] = pd.cut(
    becario_maestria["EDADBASES"], bins=bins, labels=labels, right=True
)

# 2. Calcular la tasa de participación (%) en áreas priorizadas (donde la variable binaria es 'SI')
# Asumiendo que la columna binaria se llama 'AREA_PRIORIZADA_BIN'
tasa_participacion = (
    becario_maestria.groupby(["RANGO_EDAD", "SEXO"])["AREA_PRIORIZADA_BIN"]
    .apply(lambda x: (x == "SI").mean() * 100)
    .unstack()
)

print("Tasa de Participación en Áreas Priorizadas (%) por Edad y Género:")
print(tasa_participacion.round(2))

# Configuración de estilo
sns.set_style("white")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"

# Preparar datos en formato largo (long format) para Seaborn
df_plot = (
    becario_maestria.groupby(["RANGO_EDAD", "SEXO"])["AREA_PRIORIZADA_BIN"]
    .apply(lambda x: (x == "SI").mean() * 100)
    .reset_index(name="TASA_PARTICIPACION")
)

# Crear la figura
fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)

# Graficar barras agrupadas
palette_gender = {"MASCULINO": "#2B5C8F", "FEMENINO": "#D95F02"}

sns_bar = sns.barplot(
    data=df_plot,
    x="RANGO_EDAD",
    y="TASA_PARTICIPACION",
    hue="SEXO",
    palette=palette_gender,
    ax=ax,
    width=0.5,
    edgecolor="white",
    linewidth=1,
)

# Añadir anotaciones de porcentaje sobre cada barra
for p in ax.patches:
    height = p.get_height()
    if not np.isnan(height) and height > 0:
        ax.annotate(
            f"{height:.1f}%",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color="#333333",
            xytext=(0, 4),
            textcoords="offset points",
        )

# Personalización de límites y etiquetas
ax.set_ylim(0, max(df_plot["TASA_PARTICIPACION"].fillna(0)) * 1.25)
ax.set_ylabel("Tasa de Participación (%)", fontsize=10, labelpad=10)
ax.set_xlabel("Rango de edades", fontsize=10, labelpad=10)
ax.set_title(
    "Tasa de Participación en Áreas Priorizadas por Edad y Género",
    fontsize=12,
    fontweight="bold",
    pad=15,
)

# Grilla horizontal y despined
ax.yaxis.grid(True, linestyle="--", alpha=0.3, color="#CCCCCC")
ax.set_axisbelow(True)
sns.despine(top=True, right=True)

# Leyenda compacta
ax.legend(
    title="Género",
    loc="upper right",
    frameon=True,
    facecolor="#F8F9FA",
    edgecolor="none",
    fontsize=9,
    title_fontsize=9,
)

plt.tight_layout()
plt.show()

###############################################################################
# Se analiza la evolución temporal de cada área priorizada
###############################################################################

# Configuración de estilo académico
sns.set_style("white")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"

# 1. Calcular la tabla de contingencia porcentual (Normalizada por fila/año)
col_anio = "AÑO_CONVOCATORIA"  # Reemplazar si el nombre exacto de la columna difiere
col_area = "AREA_PRIORIZADA"

# Frecuencia relativa al 100% por año
df_relativo = (
    pd.crosstab(
        becario_maestria[col_anio], becario_maestria[col_area], normalize="index"
    )
    * 100
)

# 2. Filtrar únicamente las áreas priorizadas específicas (excluyendo 'Otros' o 'No Priorizado')
areas_especificas = [
    col for col in df_relativo.columns if col not in ["Otros"]
]
df_plot = df_relativo[areas_especificas]

# 3. Construir el gráfico de líneas de evolución
fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

# Paleta discreta de alto contraste para las líneas
paleta = sns.color_palette("Set1", n_colors=len(areas_especificas))

for idx, area in enumerate(areas_especificas):
    ax.plot(
        df_plot.index.astype(str),
        df_plot[area],
        marker="o",
        linewidth=2.2,
        markersize=6,
        label=area,
        color=paleta[idx],
    )

    # Anotar los valores clave en los puntos de la línea
    for x_idx, (anio, val) in enumerate(df_plot[area].items()):
        if val > 0:  # Mostrar únicamente si hay presencia
            ax.annotate(
                f"{val:.1f}%",
                (x_idx, val),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=8,
                fontweight="bold",
                color=paleta[idx],
            )

# Estética y ejes
ax.set_ylabel(
    "Proporción del Total Anual (%)", fontsize=10, labelpad=10
)
ax.set_xlabel("Año de Convocatoria", fontsize=10, labelpad=10)
ax.set_title(
    "Evolución de la Participación Relativa por Área Priorizada (2013-2025)",
    fontsize=12,
    fontweight="bold",
    pad=15,
)

# Ajuste dinámico del límite vertical para dar espacio a las anotaciones
max_y = df_plot.max().max()
ax.set_ylim(0, max_y * 1.25)

# Grilla y despinado
ax.yaxis.grid(True, linestyle="--", alpha=0.3, color="#CCCCCC")
ax.set_axisbelow(True)
sns.despine(top=True, right=True)

# Leyenda
ax.legend(
    title="Área Priorizada",
    loc="upper right",
    frameon=True,
    facecolor="#F8F9FA",
    edgecolor="none",
    fontsize=9,
    title_fontsize=9,
)

plt.tight_layout()
plt.show()












