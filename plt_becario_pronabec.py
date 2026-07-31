# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 11:26:45 2026

@author: Enzo
"""

###############################################################################
# PROYECTO PARA ELABORAR UNA CARACTERIZACIÓN DE BECARIOS DEL PRONABEC
###############################################################################

###############################################################################
# OBJETIVO: ESTE PROYECTO REPRESENTA UNA REFORMULACIÓN DE LA CARACTERIZACIÓN
# REALIZADA PREVIAMENTE
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

# Se identifica la distribución de maestria y doctorado considerando el dataframe pronabec
pronabec.NIVEL_EDUCATIVO.value_counts()

# Se eliminan estos registros
pronabec = pronabec[pronabec["NIVEL_EDUCATIVO"]!="SIN REGISTRO"]

# Se analiza detalladamente la columna CONDICION_FINAL del dataframe pronabec
pronabec["CONDICION_FINAL"].head(10)
pronabec.CONDICION_FINAL.value_counts()
pronabec.CONDICION_FINAL.value_counts(normalize=True)

# Se identifica la presencia de nulos en la columna CONDICION_FINAL del dataframe pronabec
nulo = pronabec["CONDICION_FINAL"].isna().sum()
print(f"la columna CONDICION_FINAL contiene {nulo} valores nulos")

# Se calcula la distribución de maestria y doctorado onsidernando el dataframe pronabec
pronabec.NIVEL_EDUCATIVO.value_counts()
pronabec.NIVEL_EDUCATIVO.value_counts(normalize=True).round(2)*100

# Se analiza la distribución de postulantes por genero
pronabec.SEXO.value_counts(normalize=True).round(2)*100

# Se obtiene un grafico juntos para el nivel educativo del postulante
pronabec_postu = pd.pivot_table(pronabec, values="ID_POSTULACION", index="AÑO_CONVOCATORIA", columns="NIVEL_EDUCATIVO", aggfunc="count")
pronabec_postu.reset_index(inplace=True)

# Se reemplazan algunas columnas con valor nan a 0
pronabec_postu["DOCTORADO"] = pronabec_postu["DOCTORADO"].fillna(0)

# Se calcula un campo que representa la suma entre los niveles educativos
pronabec_postu["TOTAL"] = pronabec_postu["DOCTORADO"] + pronabec_postu["MAESTRIA"]

# Se convierte año convocatoria a una variable string
pronabec_postu["AÑO_CONVOCATORIA"] = pronabec_postu["AÑO_CONVOCATORIA"].astype(str)

x = range(len(pronabec_postu))

color_maestria = "#5FB7C6"
color_doctorado = "#A3AD2C"
color_total = "#0B4F6C"

fig, ax = plt.subplots(figsize=(13, 6))

# =====================
# Barras apiladas
# Maestría abajo, Doctorado arriba
# =====================
b_maestria = ax.bar(
    x,
    pronabec_postu["MAESTRIA"],
    color=color_maestria,
    width=0.72,
    label="Maestría"
)

b_doctorado = ax.bar(
    x,
    pronabec_postu["DOCTORADO"],
    bottom=pronabec_postu["MAESTRIA"],
    color=color_doctorado,
    width=0.72,
    label="Doctorado"
)

# =====================
# Etiquetas Maestría
# =====================
for i, mae in enumerate(pronabec_postu["MAESTRIA"]):
    if mae >= 80:
        ax.text(
            i,
            mae / 2,
            f"{mae:,.0f}",
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color="white"
        )

# =====================
# Etiquetas Doctorado
# Se colocan encima del segmento para que no se pierdan
# =====================
for i, (mae, doc) in enumerate(zip(pronabec_postu["MAESTRIA"], pronabec_postu["DOCTORADO"])):
    if doc > 0:
        ax.text(
            i,
            mae + doc + 18,
            f"{doc:,.0f}",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
            color=color_doctorado
        )

# =====================
# Etiquetas Total
# =====================
altura_max = pronabec_postu["TOTAL"].max()

for i, total in enumerate(pronabec_postu["TOTAL"]):
    ax.text(
        i,
        total + altura_max * 0.080,
        f"{total:,.0f}",
        ha="center",
        va="bottom",
        fontsize=14,
        fontweight="bold",
        color=color_total,
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            edgecolor=color_total,
            linewidth=1.2
        )
    )

# =====================
# Formato general
# =====================
#ax.set_title(
    #"Evolución de becarios según nivel de formación académica",
    #fontsize=15,
    #fontweight="bold",
    #pad=18
#)

ax.set_xlabel("Año")
ax.set_ylabel("Número de postulantes")

ax.set_xticks(x)
ax.set_xticklabels(pronabec_postu["AÑO_CONVOCATORIA"])

ax.yaxis.set_major_formatter(
    mticker.StrMethodFormatter("{x:,.0f}")
)

ax.set_ylim(0, pronabec_postu["TOTAL"].max() * 1.15)

ax.grid(axis="y", linestyle="--", alpha=0.25)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(
    frameon=False,
    ncol=2,
    loc="upper right"
)

plt.tight_layout()
plt.show()


###############################################################################
# Se analiza los postulantes a programas de maestria
###############################################################################
postulante_maestria = pronabec[pronabec["NIVEL_EDUCATIVO"]=="MAESTRIA"]

# Se calcula la distribución de postulantes por año
postulante_maestria_año = pd.pivot_table(postulante_maestria, values="ID_POSTULACION", index="AÑO_CONVOCATORIA", aggfunc="count")
postulante_maestria_año.reset_index(inplace=True)

# Se transforman campos del dataframe pronabec_becario_año
postulante_maestria_año["AÑO_CONVOCATORIA"] = pd.to_numeric(postulante_maestria_año["AÑO_CONVOCATORIA"], errors="coerce")
postulante_maestria_año["ID_POSTULACION"] = pd.to_numeric(postulante_maestria_año["ID_POSTULACION"], errors="coerce")

# Se renombran atributos del dataframe pronabec_becario_año
postulante_maestria_año.rename(columns=({"ID_POSTULACION":"CANTIDAD"}), inplace=True)

# Se organiza en función de año
postulante_maestria_año = postulante_maestria_año.sort_values("AÑO_CONVOCATORIA")

# Se calcula el rango de edades de los postulantes de maestría
edad_resumen = (
    postulante_maestria.groupby("AÑO_CONVOCATORIA")["EDADBASES"]
      .agg(
          EDAD_MINIMA="min",
          EDAD_MEDIANA="median",
          EDAD_MAXIMA="max"
      )
      .reset_index()
)

edad_resumen


# ======================================
# Gráfico
# ======================================
fig, ax = plt.subplots(figsize=(12, 8))

# Bastones Min-Max
ax.vlines(
    edad_resumen["AÑO_CONVOCATORIA"],
    edad_resumen["EDAD_MINIMA"],
    edad_resumen["EDAD_MAXIMA"],
    color="#BFBFBF",
    linewidth=3,
    zorder=1
)

# Puntos mínimos
ax.scatter(
    edad_resumen["AÑO_CONVOCATORIA"],
    edad_resumen["EDAD_MINIMA"],
    color="#A3AD2C",
    s=60,
    label="Edad mínima",
    zorder=3
)

# Puntos medianos
ax.scatter(
    edad_resumen["AÑO_CONVOCATORIA"],
    edad_resumen["EDAD_MEDIANA"],
    color="#0B4F6C",
    s=110,
    label="Edad mediana",
    zorder=4
)

# Puntos máximos
ax.scatter(
    edad_resumen["AÑO_CONVOCATORIA"],
    edad_resumen["EDAD_MAXIMA"],
    color="#C0392B",
    s=60,
    label="Edad máxima",
    zorder=3
)

# ======================================
# Etiquetas
# ======================================
for _, row in edad_resumen.iterrows():

    # Edad mínima
    ax.annotate(
        f'{row["EDAD_MINIMA"]:.0f}',
        (row["AÑO_CONVOCATORIA"], row["EDAD_MINIMA"]),
        xytext=(-12, -15),
        textcoords="offset points",
        fontsize=16,
        color="#A3AD2C",
        fontweight="bold"
    )

    # Edad mediana
    ax.annotate(
        f'{row["EDAD_MEDIANA"]:.0f}',
        (row["AÑO_CONVOCATORIA"], row["EDAD_MEDIANA"]),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        fontsize=16,
        color="#0B4F6C",
        fontweight="bold"
    )

    # Edad máxima
    ax.annotate(
        f'{row["EDAD_MAXIMA"]:.0f}',
        (row["AÑO_CONVOCATORIA"], row["EDAD_MAXIMA"]),
        xytext=(10, 6),
        textcoords="offset points",
        fontsize=16,
        color="#C0392B",
        fontweight="bold"
    )

# ======================================
# Formato
# ======================================
ax.set_xlabel("Año")
ax.set_ylabel("Edad (años)")

# Mostrar todos los años
ax.set_xticks(edad_resumen["AÑO_CONVOCATORIA"])
ax.set_xticklabels(
    edad_resumen["AÑO_CONVOCATORIA"],
    rotation=0
)

# Espacio para etiquetas
ax.set_ylim(
    edad_resumen["EDAD_MINIMA"].min() - 2,
    edad_resumen["EDAD_MAXIMA"].max() + 4
)

# Estética
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)

ax.legend(
    frameon=False,
    ncol=3,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.08),
    fontsize=16
)

plt.tight_layout()
plt.show()


# Se calcula el sexo de los postulantes de maestría
sexo_total = (
    postulante_maestria
    .groupby(["AÑO_CONVOCATORIA", "SEXO"])
    .size()
    .reset_index(name="TOTAL")
)

sexo_total["PROPORCION (%)"] = (
    sexo_total.groupby("AÑO_CONVOCATORIA")["TOTAL"]
    .transform(lambda x: round(x / x.sum() * 100, 1))
)

sexo_total

sexo_pivot = sexo_total.pivot(
    index="AÑO_CONVOCATORIA",
    columns="SEXO",
    values="PROPORCION (%)"
).fillna(0)

sexo_pivot

fig, ax = plt.subplots(figsize=(12, 6))

ax.bar(
    sexo_pivot.index,
    sexo_pivot["FEMENINO"],
    color="#C0392B",
    label="Femenino"
)

ax.bar(
    sexo_pivot.index,
    sexo_pivot["MASCULINO"],
    bottom=sexo_pivot["FEMENINO"],
    color="#0B4F6C",
    label="Masculino"
)

# Etiquetas
for i, año in enumerate(sexo_pivot.index):

    fem = sexo_pivot.loc[año, "FEMENINO"]
    masc = sexo_pivot.loc[año, "MASCULINO"]

    ax.text(
        año,
        fem/2,
        f"{fem:.1f}%",
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
        fontsize=12
    )

    ax.text(
        año,
        fem + masc/2,
        f"{masc:.1f}%",
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
        fontsize=12
    )

ax.set_ylabel("Participación (%)")
ax.set_xlabel("Año")
ax.set_ylim(0, 100)

ax.set_xticks(sexo_pivot.index)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.grid(axis="y", linestyle="--", alpha=0.3)

ax.legend(
    frameon=False,
    ncol=2,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    fontsize=14    
)

plt.tight_layout()
plt.show()

###############################################################################
# Se analiza los postulantes a programas de doctorado
###############################################################################
postulante_doctorado = pronabec[pronabec["NIVEL_EDUCATIVO"]=="DOCTORADO"]

# Se calcula la distribución de postulantes por año
postulante_doctorado_año = pd.pivot_table(postulante_doctorado, values="ID_POSTULACION", index="AÑO_CONVOCATORIA", aggfunc="count")
postulante_doctorado_año.reset_index(inplace=True)

# Se transforman campos del dataframe pronabec_becario_año
postulante_doctorado_año["AÑO_CONVOCATORIA"] = pd.to_numeric(postulante_doctorado_año["AÑO_CONVOCATORIA"], errors="coerce")
postulante_doctorado_año["ID_POSTULACION"] = pd.to_numeric(postulante_doctorado_año["ID_POSTULACION"], errors="coerce")

# Se renombran atributos del dataframe pronabec_becario_año
postulante_doctorado_año.rename(columns=({"ID_POSTULACION":"CANTIDAD"}), inplace=True)

# Se organiza en función de año
postulante_doctorado_año = postulante_doctorado_año.sort_values("AÑO_CONVOCATORIA")

# Se calcula el rango de edades de los postulantes de maestría
edad_resumen = (
    postulante_doctorado.groupby("AÑO_CONVOCATORIA")["EDADBASES"]
      .agg(
          EDAD_MINIMA="min",
          EDAD_MEDIANA="median",
          EDAD_MAXIMA="max"
      )
      .reset_index()
)

edad_resumen


# ======================================
# Gráfico
# ======================================
fig, ax = plt.subplots(figsize=(12, 8))

# Bastones Min-Max
ax.vlines(
    edad_resumen["AÑO_CONVOCATORIA"],
    edad_resumen["EDAD_MINIMA"],
    edad_resumen["EDAD_MAXIMA"],
    color="#BFBFBF",
    linewidth=3,
    zorder=1
)

# Puntos mínimos
ax.scatter(
    edad_resumen["AÑO_CONVOCATORIA"],
    edad_resumen["EDAD_MINIMA"],
    color="#A3AD2C",
    s=60,
    label="Edad mínima",
    zorder=3
)

# Puntos medianos
ax.scatter(
    edad_resumen["AÑO_CONVOCATORIA"],
    edad_resumen["EDAD_MEDIANA"],
    color="#0B4F6C",
    s=110,
    label="Edad mediana",
    zorder=4
)

# Puntos máximos
ax.scatter(
    edad_resumen["AÑO_CONVOCATORIA"],
    edad_resumen["EDAD_MAXIMA"],
    color="#C0392B",
    s=60,
    label="Edad máxima",
    zorder=3
)

# ======================================
# Etiquetas
# ======================================
for _, row in edad_resumen.iterrows():

    # Edad mínima
    ax.annotate(
        f'{row["EDAD_MINIMA"]:.0f}',
        (row["AÑO_CONVOCATORIA"], row["EDAD_MINIMA"]),
        xytext=(-12, -15),
        textcoords="offset points",
        fontsize=16,
        color="#A3AD2C",
        fontweight="bold"
    )

    # Edad mediana
    ax.annotate(
        f'{row["EDAD_MEDIANA"]:.0f}',
        (row["AÑO_CONVOCATORIA"], row["EDAD_MEDIANA"]),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        fontsize=16,
        color="#0B4F6C",
        fontweight="bold"
    )

    # Edad máxima
    ax.annotate(
        f'{row["EDAD_MAXIMA"]:.0f}',
        (row["AÑO_CONVOCATORIA"], row["EDAD_MAXIMA"]),
        xytext=(10, 6),
        textcoords="offset points",
        fontsize=16,
        color="#C0392B",
        fontweight="bold"
    )

# ======================================
# Formato
# ======================================
ax.set_xlabel("Año")
ax.set_ylabel("Edad (años)")

# Mostrar todos los años
ax.set_xticks(edad_resumen["AÑO_CONVOCATORIA"])
ax.set_xticklabels(
    edad_resumen["AÑO_CONVOCATORIA"],
    rotation=0
)

# Espacio para etiquetas
ax.set_ylim(
    edad_resumen["EDAD_MINIMA"].min() - 2,
    edad_resumen["EDAD_MAXIMA"].max() + 4
)

# Estética
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)

ax.legend(
    frameon=False,
    ncol=3,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.08),
    fontsize=16
)

plt.tight_layout()
plt.show()


# Se calcula el sexo de los postulantes de doctorado
sexo_total = (
    postulante_maestria
    .groupby(["AÑO_CONVOCATORIA", "SEXO"])
    .size()
    .reset_index(name="TOTAL")
)

sexo_total["PROPORCION (%)"] = (
    sexo_total.groupby("AÑO_CONVOCATORIA")["TOTAL"]
    .transform(lambda x: round(x / x.sum() * 100, 1))
)

sexo_total

sexo_pivot = sexo_total.pivot(
    index="AÑO_CONVOCATORIA",
    columns="SEXO",
    values="PROPORCION (%)"
).fillna(0)

sexo_pivot

fig, ax = plt.subplots(figsize=(12, 6))

ax.bar(
    sexo_pivot.index,
    sexo_pivot["FEMENINO"],
    color="#C0392B",
    label="Femenino"
)

ax.bar(
    sexo_pivot.index,
    sexo_pivot["MASCULINO"],
    bottom=sexo_pivot["FEMENINO"],
    color="#0B4F6C",
    label="Masculino"
)

# Etiquetas
for i, año in enumerate(sexo_pivot.index):

    fem = sexo_pivot.loc[año, "FEMENINO"]
    masc = sexo_pivot.loc[año, "MASCULINO"]

    ax.text(
        año,
        fem/2,
        f"{fem:.1f}%",
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
        fontsize=12
    )

    ax.text(
        año,
        fem + masc/2,
        f"{masc:.1f}%",
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
        fontsize=12
    )

ax.set_ylabel("Participación (%)")
ax.set_xlabel("Año")
ax.set_ylim(0, 100)

ax.set_xticks(sexo_pivot.index)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.grid(axis="y", linestyle="--", alpha=0.3)

ax.legend(
    frameon=False,
    ncol=2,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02)
)

plt.tight_layout()
plt.show()


# Se calcula el sexo de los postulantes de maestría
sexo_total = (
    postulante_doctorado
    .groupby(["AÑO_CONVOCATORIA", "SEXO"])
    .size()
    .reset_index(name="TOTAL")
)

sexo_total["PROPORCION (%)"] = (
    sexo_total.groupby("AÑO_CONVOCATORIA")["TOTAL"]
    .transform(lambda x: round(x / x.sum() * 100, 1))
)

sexo_total

sexo_pivot = sexo_total.pivot(
    index="AÑO_CONVOCATORIA",
    columns="SEXO",
    values="PROPORCION (%)"
).fillna(0)

sexo_pivot

fig, ax = plt.subplots(figsize=(12, 6))

ax.bar(
    sexo_pivot.index,
    sexo_pivot["FEMENINO"],
    color="#C0392B",
    label="Femenino"
)

ax.bar(
    sexo_pivot.index,
    sexo_pivot["MASCULINO"],
    bottom=sexo_pivot["FEMENINO"],
    color="#0B4F6C",
    label="Masculino"
)

# Etiquetas
for i, año in enumerate(sexo_pivot.index):

    fem = sexo_pivot.loc[año, "FEMENINO"]
    masc = sexo_pivot.loc[año, "MASCULINO"]

    ax.text(
        año,
        fem/2,
        f"{fem:.1f}%",
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
        fontsize=12
    )

    ax.text(
        año,
        fem + masc/2,
        f"{masc:.1f}%",
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
        fontsize=12
    )

ax.set_ylabel("Participación (%)")
ax.set_xlabel("Año")
ax.set_ylim(0, 100)

ax.set_xticks(sexo_pivot.index)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.grid(axis="y", linestyle="--", alpha=0.3)

ax.legend(
    frameon=False,
    ncol=2,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    fontsize=14    
)

plt.tight_layout()
plt.show()

###############################################################################
# Se analizan a los becarios, es decir, postulantes que recibieron la beca
###############################################################################

becario = pronabec[pronabec["CONDICION_FINAL"]=="SE LE ADJUDICÓ LA BECA"]

# Se cuentan los valores únicos del dataframe becario
becario["ID_POSTULACION"].nunique()

# Se analia la distribución de nivel de educación de las observaciones que integran el dataframe becario
becario.NIVEL_EDUCATIVO.value_counts()
becario.NIVEL_EDUCATIVO.value_counts(normalize=True)


# Se renombran algunos registros de la columna REGION_PREGRADO del dataframe becario
becario.REGION_PREGRADO.value_counts()
becario["REGION_PREGRADO"] = becario["REGION_PREGRADO"].replace({
    "HUANCAYO":"JUNIN",
    "TRUJILLO":"LA LIBERTAD",
    "HUACHO":"LIMA",
    "CHACHAPOYAS":"AMAZONAS",
    "CHOTA":"CAJAMARCA",
    "TARAPOTO":"SAN MARTIN",
    "TINGO MARIA ":"HUANUCO",
    "JAÉN-CAJAMARCA":"CAJAMARCA",
    "IQUITOS":"LORETO",
    "CALLAO":"LIMA"
    })



# Se obtiene un grafico juntos para el nivel educativo del becario
becario_nivel = pd.pivot_table(becario, values="ID_POSTULACION", index="AÑO_CONVOCATORIA", columns="NIVEL_EDUCATIVO", aggfunc="count")
becario_nivel.reset_index(inplace=True)

# Se reemplazan algunas columnas con valor nan a 0
becario_nivel["DOCTORADO"] = becario_nivel["DOCTORADO"].fillna(0)

# Se calcula un campo que representa la suma entre los niveles educativos
becario_nivel["TOTAL"] = becario_nivel["DOCTORADO"] + becario_nivel["MAESTRIA"]

# Se convierte año convocatoria a una variable string
becario_nivel["AÑO_CONVOCATORIA"] = becario_nivel["AÑO_CONVOCATORIA"].astype(str)

x = range(len(becario_nivel))

color_maestria = "#5FB7C6"
color_doctorado = "#A3AD2C"
color_total = "#0B4F6C"

fig, ax = plt.subplots(figsize=(13, 6))

# =====================
# Barras apiladas
# Maestría abajo, Doctorado arriba
# =====================
b_maestria = ax.bar(
    x,
    becario_nivel["MAESTRIA"],
    color=color_maestria,
    width=0.72,
    label="Maestría"
)

b_doctorado = ax.bar(
    x,
    becario_nivel["DOCTORADO"],
    bottom=becario_nivel["MAESTRIA"],
    color=color_doctorado,
    width=0.72,
    label="Doctorado"
)

# =====================
# Etiquetas Maestría
# =====================
for i, mae in enumerate(becario_nivel["MAESTRIA"]):
    if mae >= 80:
        ax.text(
            i,
            mae / 2,
            f"{mae:,.0f}",
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color="white"
        )

# =====================
# Etiquetas Doctorado
# Se colocan encima del segmento para que no se pierdan
# =====================
for i, (mae, doc) in enumerate(zip(becario_nivel["MAESTRIA"], becario_nivel["DOCTORADO"])):
    if doc > 0:
        ax.text(
            i,
            mae + doc + 18,
            f"{doc:,.0f}",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
            color=color_doctorado
        )

# =====================
# Etiquetas Total
# =====================
altura_max = pronabec_postu["TOTAL"].max()

for i, total in enumerate(becario_nivel["TOTAL"]):
    ax.text(
        i,
        total + altura_max * 0.050,
        f"{total:,.0f}",
        ha="center",
        va="bottom",
        fontsize=14,
        fontweight="bold",
        color=color_total,
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            edgecolor=color_total,
            linewidth=1.2
        )
    )

# =====================
# Formato general
# =====================
#ax.set_title(
    #"Evolución de becarios según nivel de formación académica",
    #fontsize=15,
    #fontweight="bold",
    #pad=18
#)

ax.set_xlabel("Año")
ax.set_ylabel("Número de becarios")

ax.set_xticks(x)
ax.set_xticklabels(becario_nivel["AÑO_CONVOCATORIA"])

ax.yaxis.set_major_formatter(
    mticker.StrMethodFormatter("{x:,.0f}")
)

ax.set_ylim(0, becario_nivel["TOTAL"].max() * 1.15)

ax.grid(axis="y", linestyle="--", alpha=0.25)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(
    frameon=False,
    ncol=2,
    loc="upper right"
)

plt.tight_layout()
plt.show()


###############################################################################
# Se analiza los postulantes a programas de maestria
###############################################################################
becario_maestria = becario[becario["NIVEL_EDUCATIVO"]=="MAESTRIA"]

# Se calcula la distribución de postulantes por año
becario_maestria_año = pd.pivot_table(becario_maestria, values="ID_POSTULACION", index="AÑO_CONVOCATORIA", aggfunc="count")
becario_maestria_año.reset_index(inplace=True)

# Se transforman campos del dataframe pronabec_becario_año
becario_maestria_año["AÑO_CONVOCATORIA"] = pd.to_numeric(becario_maestria_año["AÑO_CONVOCATORIA"], errors="coerce")
becario_maestria_año["ID_POSTULACION"] = pd.to_numeric(becario_maestria_año["ID_POSTULACION"], errors="coerce")

# Se renombran atributos del dataframe pronabec_becario_año
becario_maestria_año.rename(columns=({"ID_POSTULACION":"CANTIDAD"}), inplace=True)

# Se organiza en función de año
becario_maestria_año = becario_maestria_año.sort_values("AÑO_CONVOCATORIA")

# Se calcula el rango de edades de los postulantes de maestría
edad_resumen = (
    becario_maestria.groupby("AÑO_CONVOCATORIA")["EDADBASES"]
      .agg(
          EDAD_MINIMA="min",
          EDAD_MEDIANA="median",
          EDAD_MAXIMA="max"
      )
      .reset_index()
)

edad_resumen


# ======================================
# Gráfico
# ======================================
fig, ax = plt.subplots(figsize=(12, 8))

# Bastones Min-Max
ax.vlines(
    edad_resumen["AÑO_CONVOCATORIA"],
    edad_resumen["EDAD_MINIMA"],
    edad_resumen["EDAD_MAXIMA"],
    color="#BFBFBF",
    linewidth=3,
    zorder=1
)

# Puntos mínimos
ax.scatter(
    edad_resumen["AÑO_CONVOCATORIA"],
    edad_resumen["EDAD_MINIMA"],
    color="#A3AD2C",
    s=60,
    label="Edad mínima",
    zorder=3
)

# Puntos medianos
ax.scatter(
    edad_resumen["AÑO_CONVOCATORIA"],
    edad_resumen["EDAD_MEDIANA"],
    color="#0B4F6C",
    s=110,
    label="Edad mediana",
    zorder=4
)

# Puntos máximos
ax.scatter(
    edad_resumen["AÑO_CONVOCATORIA"],
    edad_resumen["EDAD_MAXIMA"],
    color="#C0392B",
    s=60,
    label="Edad máxima",
    zorder=3
)

# ======================================
# Etiquetas
# ======================================
for _, row in edad_resumen.iterrows():

    # Edad mínima
    ax.annotate(
        f'{row["EDAD_MINIMA"]:.0f}',
        (row["AÑO_CONVOCATORIA"], row["EDAD_MINIMA"]),
        xytext=(-12, -15),
        textcoords="offset points",
        fontsize=16,
        color="#A3AD2C",
        fontweight="bold"
    )

    # Edad mediana
    ax.annotate(
        f'{row["EDAD_MEDIANA"]:.0f}',
        (row["AÑO_CONVOCATORIA"], row["EDAD_MEDIANA"]),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        fontsize=16,
        color="#0B4F6C",
        fontweight="bold"
    )

    # Edad máxima
    ax.annotate(
        f'{row["EDAD_MAXIMA"]:.0f}',
        (row["AÑO_CONVOCATORIA"], row["EDAD_MAXIMA"]),
        xytext=(10, 6),
        textcoords="offset points",
        fontsize=16,
        color="#C0392B",
        fontweight="bold"
    )

# ======================================
# Formato
# ======================================
ax.set_xlabel("Año")
ax.set_ylabel("Edad (años)")

# Mostrar todos los años
ax.set_xticks(edad_resumen["AÑO_CONVOCATORIA"])
ax.set_xticklabels(
    edad_resumen["AÑO_CONVOCATORIA"],
    rotation=0
)

# Espacio para etiquetas
ax.set_ylim(
    edad_resumen["EDAD_MINIMA"].min() - 2,
    edad_resumen["EDAD_MAXIMA"].max() + 4
)

# Estética
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)

ax.legend(
    frameon=False,
    ncol=3,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.08),
    fontsize=16
)

plt.tight_layout()
plt.show()


# Se calcula el sexo de los postulantes de maestría
sexo_total = (
    becario_maestria
    .groupby(["AÑO_CONVOCATORIA", "SEXO"])
    .size()
    .reset_index(name="TOTAL")
)

sexo_total["PROPORCION (%)"] = (
    sexo_total.groupby("AÑO_CONVOCATORIA")["TOTAL"]
    .transform(lambda x: round(x / x.sum() * 100, 1))
)

sexo_total

sexo_pivot = sexo_total.pivot(
    index="AÑO_CONVOCATORIA",
    columns="SEXO",
    values="PROPORCION (%)"
).fillna(0)

sexo_pivot

fig, ax = plt.subplots(figsize=(12, 6))

ax.bar(
    sexo_pivot.index,
    sexo_pivot["FEMENINO"],
    color="#C0392B",
    label="Femenino"
)

ax.bar(
    sexo_pivot.index,
    sexo_pivot["MASCULINO"],
    bottom=sexo_pivot["FEMENINO"],
    color="#0B4F6C",
    label="Masculino"
)

# Etiquetas
for i, año in enumerate(sexo_pivot.index):

    fem = sexo_pivot.loc[año, "FEMENINO"]
    masc = sexo_pivot.loc[año, "MASCULINO"]

    ax.text(
        año,
        fem/2,
        f"{fem:.1f}%",
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
        fontsize=12
    )

    ax.text(
        año,
        fem + masc/2,
        f"{masc:.1f}%",
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
        fontsize=12
    )

ax.set_ylabel("Participación (%)")
ax.set_xlabel("Año")
ax.set_ylim(0, 100)

ax.set_xticks(sexo_pivot.index)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.grid(axis="y", linestyle="--", alpha=0.3)

ax.legend(
    frameon=False,
    ncol=2,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    fontsize=14    
)

plt.tight_layout()
plt.show()


###############################################################################
# Se analiza a los becarios a programas de doctorado
###############################################################################
becario_doctorado = becario[becario["NIVEL_EDUCATIVO"]=="DOCTORADO"]

# Se calcula la distribución de postulantes por año
becario_doctorado_año = pd.pivot_table(becario_doctorado, values="ID_POSTULACION", index="AÑO_CONVOCATORIA", aggfunc="count")
becario_doctorado_año.reset_index(inplace=True)

# Se transforman campos del dataframe pronabec_becario_año
becario_doctorado_año["AÑO_CONVOCATORIA"] = pd.to_numeric(becario_doctorado_año["AÑO_CONVOCATORIA"], errors="coerce")
becario_doctorado_año["ID_POSTULACION"] = pd.to_numeric(becario_doctorado_año["ID_POSTULACION"], errors="coerce")

# Se renombran atributos del dataframe pronabec_becario_año
becario_doctorado_año.rename(columns=({"ID_POSTULACION":"CANTIDAD"}), inplace=True)

# Se organiza en función de año
becario_doctorado_año = becario_doctorado_año.sort_values("AÑO_CONVOCATORIA")

# Se calcula el rango de edades de los postulantes de maestría
edad_resumen = (
    becario_doctorado.groupby("AÑO_CONVOCATORIA")["EDADBASES"]
      .agg(
          EDAD_MINIMA="min",
          EDAD_MEDIANA="median",
          EDAD_MAXIMA="max"
      )
      .reset_index()
)

edad_resumen


# ======================================
# Gráfico
# ======================================
fig, ax = plt.subplots(figsize=(12, 8))

# Bastones Min-Max
ax.vlines(
    edad_resumen["AÑO_CONVOCATORIA"],
    edad_resumen["EDAD_MINIMA"],
    edad_resumen["EDAD_MAXIMA"],
    color="#BFBFBF",
    linewidth=3,
    zorder=1
)

# Puntos mínimos
ax.scatter(
    edad_resumen["AÑO_CONVOCATORIA"],
    edad_resumen["EDAD_MINIMA"],
    color="#A3AD2C",
    s=60,
    label="Edad mínima",
    zorder=3
)

# Puntos medianos
ax.scatter(
    edad_resumen["AÑO_CONVOCATORIA"],
    edad_resumen["EDAD_MEDIANA"],
    color="#0B4F6C",
    s=110,
    label="Edad mediana",
    zorder=4
)

# Puntos máximos
ax.scatter(
    edad_resumen["AÑO_CONVOCATORIA"],
    edad_resumen["EDAD_MAXIMA"],
    color="#C0392B",
    s=60,
    label="Edad máxima",
    zorder=3
)

# ======================================
# Etiquetas
# ======================================
for _, row in edad_resumen.iterrows():

    # Edad mínima
    ax.annotate(
        f'{row["EDAD_MINIMA"]:.0f}',
        (row["AÑO_CONVOCATORIA"], row["EDAD_MINIMA"]),
        xytext=(-12, -15),
        textcoords="offset points",
        fontsize=16,
        color="#A3AD2C",
        fontweight="bold"
    )

    # Edad mediana
    ax.annotate(
        f'{row["EDAD_MEDIANA"]:.0f}',
        (row["AÑO_CONVOCATORIA"], row["EDAD_MEDIANA"]),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        fontsize=16,
        color="#0B4F6C",
        fontweight="bold"
    )

    # Edad máxima
    ax.annotate(
        f'{row["EDAD_MAXIMA"]:.0f}',
        (row["AÑO_CONVOCATORIA"], row["EDAD_MAXIMA"]),
        xytext=(10, 6),
        textcoords="offset points",
        fontsize=16,
        color="#C0392B",
        fontweight="bold"
    )

# ======================================
# Formato
# ======================================
ax.set_xlabel("Año")
ax.set_ylabel("Edad (años)")

# Mostrar todos los años
ax.set_xticks(edad_resumen["AÑO_CONVOCATORIA"])
ax.set_xticklabels(
    edad_resumen["AÑO_CONVOCATORIA"],
    rotation=0
)

# Espacio para etiquetas
ax.set_ylim(
    edad_resumen["EDAD_MINIMA"].min() - 2,
    edad_resumen["EDAD_MAXIMA"].max() + 4
)

# Estética
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)

ax.legend(
    frameon=False,
    ncol=3,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.08),
    fontsize=16
)

plt.tight_layout()
plt.show()


#Se calcula el sexo de los postulantes de doctorado
sexo_total = (
    becario_doctorado
    .groupby(["AÑO_CONVOCATORIA", "SEXO"])
    .size()
    .reset_index(name="TOTAL")
)

sexo_total["PROPORCION (%)"] = (
    sexo_total.groupby("AÑO_CONVOCATORIA")["TOTAL"]
    .transform(lambda x: round(x / x.sum() * 100, 1))
)

sexo_total

sexo_pivot = sexo_total.pivot(
    index="AÑO_CONVOCATORIA",
    columns="SEXO",
    values="PROPORCION (%)"
).fillna(0)

sexo_pivot

fig, ax = plt.subplots(figsize=(12, 6))

ax.bar(
    sexo_pivot.index,
    sexo_pivot["FEMENINO"],
    color="#C0392B",
    label="Femenino"
)

ax.bar(
    sexo_pivot.index,
    sexo_pivot["MASCULINO"],
    bottom=sexo_pivot["FEMENINO"],
    color="#0B4F6C",
    label="Masculino"
)

# Etiquetas
for i, año in enumerate(sexo_pivot.index):

    fem = sexo_pivot.loc[año, "FEMENINO"]
    masc = sexo_pivot.loc[año, "MASCULINO"]

    ax.text(
        año,
        fem/2,
        f"{fem:.1f}%",
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
        fontsize=12
    )

    ax.text(
        año,
        fem + masc/2,
        f"{masc:.1f}%",
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
        fontsize=12
    )

ax.set_ylabel("Participación (%)")
ax.set_xlabel("Año")
ax.set_ylim(0, 100)

ax.set_xticks(sexo_pivot.index)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.grid(axis="y", linestyle="--", alpha=0.3)

ax.legend(
    frameon=False,
    ncol=2,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    fontsize=14    
)

plt.tight_layout()
plt.show()


####################################################################################
# Se analiza el pais de destino de los becarios de programas de maestria y doctorado
# considerando los dataframes becario_maestria y becario_doctorado
####################################################################################

# Se realiza el análisis para becarios de programas de maestría
total = becario_maestria["PAISDESTINO"].value_counts()

proporcion = (
    becario_maestria["PAISDESTINO"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

pais_maestria = pd.DataFrame({
    "TOTAL": total,
    "PROPORCION (%)": proporcion
})

pais_maestria
pais_maestria.reset_index(inplace=True)
pais_maestria = pais_maestria.head(10)
pais_maestria.columns

pais_maestria = pais_maestria.sort_values("TOTAL", ascending=True)

color_principal = "#0B4F6C"   # azul institucional
color_secundario = "#5FB7C6"  # celeste

plt.figure(figsize=(12, 8))

bars = plt.barh(
    pais_maestria["PAISDESTINO"],
    pais_maestria["PROPORCION (%)"],
    color=color_principal
)

# Etiquetas: porcentaje + total
for i, (pct, total) in enumerate(
    zip(
        pais_maestria["PROPORCION (%)"],
        pais_maestria["TOTAL"]
    )
):
    plt.text(
        pct + 0.3,
        i,
        f"{pct:.1f}% ({total:,})",
        va="center",
        fontsize=16
    )

plt.xlabel("Participación (%)")

plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

plt.grid(axis="x", linestyle="--", alpha=0.3)

plt.tight_layout()
plt.show()


# Se realiza el análisis para becarios de programas de doctorado
total = becario_doctorado["PAISDESTINO"].value_counts()

proporcion = (
    becario_doctorado["PAISDESTINO"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

pais_doctorado = pd.DataFrame({
    "TOTAL": total,
    "PROPORCION (%)": proporcion
})

pais_doctorado
pais_doctorado.reset_index(inplace=True)
pais_doctorado = pais_doctorado.head(10)
pais_doctorado.columns

pais_doctorado = pais_doctorado.sort_values("TOTAL", ascending=True)

color_principal = "#0B4F6C"   # azul institucional
color_secundario = "#5FB7C6"  # celeste

plt.figure(figsize=(12, 8))

bars = plt.barh(
    pais_doctorado["PAISDESTINO"],
    pais_doctorado["PROPORCION (%)"],
    color=color_principal
)

# Etiquetas: porcentaje + total
for i, (pct, total) in enumerate(
    zip(
        pais_doctorado["PROPORCION (%)"],
        pais_doctorado["TOTAL"]
    )
):
    plt.text(
        pct + 0.3,
        i,
        f"{pct:.1f}% ({total:,})",
        va="center",
        fontsize=16
    )

plt.xlabel("Participación (%)")

plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

plt.grid(axis="x", linestyle="--", alpha=0.3)

plt.tight_layout()
plt.show()

###############################################################################
# Se analiza la institución considerando los dataframes becario_maestria y
# becario_doctorado
###############################################################################

# Se realiza el análisis para becarios de programas de maestría
total = becario_maestria["INSTITUCION"].value_counts()

proporcion = (
    becario_maestria["INSTITUCION"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

institucion_maestria = pd.DataFrame({
    "TOTAL": total,
    "PROPORCION (%)": proporcion
})

institucion_maestria
institucion_maestria.reset_index(inplace=True)
institucion_maestria = institucion_maestria.head(10)
institucion_maestria.columns

#institucion_maestria = institucion_maestria.sort_values("TOTAL", ascending=True)

color_principal = "#0B4F6C"   # azul institucional
color_destacado = "#6B8E23"   # granate

colores = [
    color_destacado
    if uni in [
        "PONTIFICIA UNIVERSIDAD CATOLICA DEL PERU",
        "UNIVERSIDAD PRIVADA SAN IGNACIO DE LOYOLA"
    ]
    else color_principal
    for uni in institucion_maestria["INSTITUCION"]
]


plt.figure(figsize=(12, 8))

bars = plt.barh(
    institucion_maestria["INSTITUCION"],
    institucion_maestria["PROPORCION (%)"],
    color=colores
)

for i, (pct, total) in enumerate(
    zip(
        institucion_maestria["PROPORCION (%)"],
        institucion_maestria["TOTAL"]
    )
):
    plt.text(
        pct + 0.3,
        i,
        f"{pct:.1f}% ({total:,})",
        va="center",
        fontsize=16
    )

plt.xlabel("Participación (%)")

plt.gca().invert_yaxis()
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

plt.grid(axis="x", linestyle="--", alpha=0.3)

plt.tight_layout()
plt.show()

###############################################################################
# Construyo un campo adicional para mi dataframe becario_maestria
###############################################################################

universidades_peruanas = [
    "PONTIFICIA UNIVERSIDAD CATOLICA DEL PERU",
    "UNIVERSIDAD PERUANA CAYETANO HEREDIA",
    "UNIVERSIDAD PRIVADA SAN IGNACIO DE LOYOLA",
    "UNIVERSIDAD DE PIURA"
]

becario_maestria["ANALISIS_INSTITUCION"] = (
    becario_maestria["INSTITUCION"]
    .str.upper()
    .isin(universidades_peruanas)
    .map({True: "PERUANA", False: "NO PERUANA"})
)


jajaja = becario_maestria[becario_maestria["ANALISIS_INSTITUCION"]=="PERUANA"]
jajaja.INSTITUCION.value_counts()


ana_total = (
    becario_maestria
    .groupby(["AÑO_CONVOCATORIA", "ANALISIS_INSTITUCION"])
    .size()
    .reset_index(name="TOTAL")
)

ana_total["PROPORCION (%)"] = (
    ana_total.groupby("AÑO_CONVOCATORIA")["TOTAL"]
    .transform(lambda x: round(x / x.sum() * 100, 1))
)

ana_total

ana_pivot = ana_total.pivot(
    index="AÑO_CONVOCATORIA",
    columns="ANALISIS_INSTITUCION",
    values="PROPORCION (%)"
).fillna(0)

ana_pivot

fig, ax = plt.subplots(figsize=(12, 6))

ax.bar(
    ana_pivot.index,
    ana_pivot["NO PERUANA"],
    color="#C0392B",
    label="No peruana"
)

ax.bar(
    ana_pivot.index,
    ana_pivot["PERUANA"],
    bottom=ana_pivot["NO PERUANA"],
    color="#0B4F6C",
    label="Peruana"
)

# Etiquetas solo si el valor es mayor a 0
for año in ana_pivot.index:

    no_peruana = ana_pivot.loc[año, "NO PERUANA"]
    peruana = ana_pivot.loc[año, "PERUANA"]

    if no_peruana > 0:
        ax.text(
            año,
            no_peruana / 2,
            f"{no_peruana:.1f}%",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=10
        )

    if peruana > 0:
        ax.text(
            año,
            no_peruana + peruana / 2,
            f"{peruana:.1f}%",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=10
        )

ax.set_ylabel("Participación (%)")
ax.set_xlabel("Año")
ax.set_ylim(0, 100)

ax.set_xticks(ana_pivot.index)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.grid(axis="y", linestyle="--", alpha=0.3)

ax.legend(
    frameon=False,
    ncol=2,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02)
)

plt.tight_layout()
plt.show()


# Se realiza el análisis para becarios de programas de doctorado
total = becario_doctorado["INSTITUCION"].value_counts()

proporcion = (
    becario_doctorado["INSTITUCION"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

institucion_doctorado = pd.DataFrame({
    "TOTAL": total,
    "PROPORCION (%)": proporcion
})

institucion_doctorado
institucion_doctorado.reset_index(inplace=True)
institucion_doctorado = institucion_doctorado.head(10)
institucion_doctorado.columns

#institucion_maestria = institucion_maestria.sort_values("TOTAL", ascending=True)

color_principal = "#0B4F6C"   # azul institucional
color_destacado = "#8E2C2C"   # granate

colores = [
    color_destacado
    if uni in [
        "PONTIFICIA UNIVERSIDAD CATOLICA DEL PERU",
        "UNIVERSIDAD PRIVADA SAN IGNACIO DE LOYOLA"
    ]
    else color_principal
    for uni in institucion_doctorado["INSTITUCION"]
]


plt.figure(figsize=(12, 8))

bars = plt.barh(
    institucion_doctorado["INSTITUCION"],
    institucion_doctorado["PROPORCION (%)"],
    color=colores
)

for i, (pct, total) in enumerate(
    zip(
        institucion_doctorado["PROPORCION (%)"],
        institucion_doctorado["TOTAL"]
    )
):
    plt.text(
        pct + 0.2,
        i,
        f"{pct:.1f}% ({total:,})",
        va="center",
        fontsize=16
    )

plt.xlabel("Participación (%)")

plt.gca().invert_yaxis()
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

plt.grid(axis="x", linestyle="--", alpha=0.3)

plt.tight_layout()
plt.show()

###############################################################################
# Se realiza un análisis considerando los títulos STEM
###############################################################################

# Se convierte en dataframe un archivo que contiene las carreras STEM según la
# clasificación del DHS

dhs = pd.read_excel("listado_STEM_CIP_2024.xlsx", sheet_name="STEM_CIP_2024", header=0)
dhs.shape
dhs.columns
dhs["Código CIP 2020"].nunique()

# Se analiza la columna NOMBRECARRERA del dataframe becario
becario["NOMBRECARRERA"].head(10)

# DataFrame que contiene la taxonomía oficial DHS
# Columnas:
#   "Título oficial en inglés"
#   "Título español"
df_stem = dhs.copy()

# DataFrame que contiene las carreras que quieres clasificar
# Columna:
#   "NOMBRECARRERA"
df = becario.copy()

# Se implementa una técnica de fuzzy matching, con coincidencia exacta y similitud
def limpiar(x):
    if pd.isna(x):
        return ""
    x = unidecode(str(x)).upper()
    x = re.sub(
        r"\b(MAESTRIA|MASTER|MAGISTER|MSC|DOCTORADO|DOCTORATE|PHD|"
        r"DOCTOR OF PHILOSOPHY|GENERICA)\b",
        " ", x
    )
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9 ]", " ", x)).strip()

# Catálogo oficial DHS bilingüe
catalogo = pd.concat([
    df_stem["Título oficial en inglés"],
    df_stem["Título_español"]
]).dropna().drop_duplicates().to_frame("TITULO_DHS")

catalogo["NORMALIZADO"] = catalogo["TITULO_DHS"].map(limpiar)
catalogo = catalogo[catalogo["NORMALIZADO"].ne("")].drop_duplicates("NORMALIZADO")
lista_dhs = catalogo["NORMALIZADO"].tolist()
titulo_original = dict(zip(catalogo["NORMALIZADO"], catalogo["TITULO_DHS"]))

def clasificar(carrera, umbral=60):
    carrera = limpiar(carrera)

    if not carrera:
        return pd.Series(["SIN INFORMACIÓN", None, 0])

    # Coincidencia exacta o por inclusión del título DHS
    incluidos = [x for x in lista_dhs if len(x.split()) >= 2 and x in carrera]
    if incluidos:
        match = max(incluidos, key=len)
        return pd.Series(["STEM", titulo_original[match], 100])

    # Coincidencia aproximada con el catálogo oficial
    match, puntaje, _ = process.extractOne(
        carrera, lista_dhs, scorer=fuzz.token_set_ratio
    )

    return pd.Series([
        "STEM" if puntaje >= umbral else "NO STEM",
        titulo_original[match],
        round(puntaje, 2)
    ])

df[["STEM", "MATCH_DHS", "PUNTAJE_MATCH"]] = (
    df["NOMBRECARRERA"].apply(clasificar)
)

df.columns
df = df[["NOMBRECARRERA","STEM", "MATCH_DHS", "PUNTAJE_MATCH"]]
df.to_excel("aproximacion_ia_carreras_stem.xlsx")

becario.to_excel("becario.xlsx")


# Considerando el archivo aproximacion_ia_carreras_stem.xlsx y el archivo becario, 
# se crea un nuevo archivo que contiene a los becarios
becarios = pd.read_excel("becario_con_STEM.xlsx", sheet_name="Sheet1", header=0)

# Considerando becarios se cuenta el universo de aplicación
becarios.STEM.value_counts()

# Se eliminan los registros que no cuentan con información
becarios = becarios[becarios["STEM"]!="SIN INFORMACIÓN"]


# Se construye un dataframe para el caso de los becarios de programas de maestria
becario_stem_maestria = becarios[becarios["NIVEL_EDUCATIVO"]=="MAESTRIA"]
becario_stem_maestria = becario_stem_maestria.STEM.value_counts(normalize=True).round(2)*100
becario_stem_maestria = becario_stem_maestria.to_frame()
becario_stem_maestria.reset_index(inplace=True)
becario_stem_maestria.rename(columns=({"proportion":"porcentaje"}), inplace=True)

labels = becario_stem_maestria["STEM"]
sizes = becario_stem_maestria["porcentaje"]

# Colores institucionales
colors = ["#0B4F6C", "#A3AD2C"]

# Crear figura
fig, ax = plt.subplots(figsize=(8, 6))

wedges, texts, autotexts = ax.pie(
    sizes,
    labels=labels,
    colors=colors,
    autopct="%1.0f%%",
    startangle=90,
    counterclock=False,
    wedgeprops={
        "edgecolor": "white",
        "linewidth": 2
    },
    textprops={
        "fontsize": 14,
        "fontweight": "bold"
    }
)

# Estilo de porcentajes
for autotext in autotexts:
    autotext.set_color("white")
    autotext.set_fontsize(14)
    autotext.set_fontweight("bold")

# Título
#ax.set_title(
    #"Distribución de programas STEM y No STEM",
    #fontsize=15,
    #fontweight="bold",
    #pad=20
#)

# Mantener círculo perfecto
ax.axis("equal")

plt.tight_layout()
plt.show()



# Se construye un dataframe para el caso de los becarios de programas de doctorado
becario_stem_doctorado = becarios[becarios["NIVEL_EDUCATIVO"]=="DOCTORADO"]
becario_stem_doctorado = becario_stem_doctorado.STEM.value_counts(normalize=True).round(2)*100
becario_stem_doctorado = becario_stem_doctorado.to_frame()
becario_stem_doctorado.reset_index(inplace=True)
becario_stem_doctorado.rename(columns=({"proportion":"porcentaje"}), inplace=True)

labels = becario_stem_doctorado["STEM"]
sizes = becario_stem_doctorado["porcentaje"]

# Colores institucionales
colors = ["#0B4F6C", "#A3AD2C"]

# Crear figura
fig, ax = plt.subplots(figsize=(8, 6))

wedges, texts, autotexts = ax.pie(
    sizes,
    labels=labels,
    colors=colors,
    autopct="%1.0f%%",
    startangle=90,
    counterclock=False,
    wedgeprops={
        "edgecolor": "white",
        "linewidth": 2
    },
    textprops={
        "fontsize": 14,
        "fontweight": "bold"
    }
)

# Estilo de porcentajes
for autotext in autotexts:
    autotext.set_color("white")
    autotext.set_fontsize(14)
    autotext.set_fontweight("bold")

# Título
#ax.set_title(
    #"Distribución de programas STEM y No STEM",
    #fontsize=15,
    #fontweight="bold",
    #pad=20
#)

# Mantener círculo perfecto
ax.axis("equal")

plt.tight_layout()
plt.show()


###############################################################################
# Se analiza el cumplimiento del compomiso de servicio (CSP), de forma global, por
# maestria y doctorado
###############################################################################

# Tabla resumen
csp = (
    becario["ESTADO_DE_CSP"]
    .value_counts()
    .reset_index()
)

csp.columns = ["ESTADO_DE_CSP", "cantidad"]

# Porcentajes
csp["porcentaje"] = (
    csp["cantidad"] /
    csp["cantidad"].sum() * 100
).round(1)

# Etiquetas amigables
csp["estado_mostrar"] = csp["ESTADO_DE_CSP"].replace({
    "CUMPLIO": "Cumplió",
    "PENDIENTE": "Pendiente",
    "SIN INFORMACIÓN": "Sin información",
    "INCUMPLIMIENTO": "Incumplimiento"
})

# Etiquetas con frecuencia
csp["label"] = (
    csp["estado_mostrar"]
    + "\n(n="
    + csp["cantidad"].map("{:,}".format)
    + ")"
)

labels = csp["label"]
sizes = csp["porcentaje"]

# Colores institucionales
colors = [
    "#0B4F6C",  # Azul petróleo
    "#5FB7C6",  # Celeste institucional
    "#A3AD2C",  # Verde oliva
    "#D9D9D9"   # Gris neutro
]

# Mostrar porcentaje solo si es >= 1%
def formato_pct(pct):
    return f"{pct:.1f}%" if pct >= 1 else ""

# Figura
fig, ax = plt.subplots(figsize=(8, 6))

wedges, texts, autotexts = ax.pie(
    sizes,
    labels=labels,
    colors=colors,
    autopct=formato_pct,
    startangle=90,
    counterclock=False,
    wedgeprops={
        "edgecolor": "white",
        "linewidth": 2
    },
    textprops={
        "fontsize": 11,
        "fontweight": "bold"
    }
)

# Estilo porcentajes
for autotext in autotexts:
    autotext.set_color("white")
    autotext.set_fontsize(12)
    autotext.set_fontweight("bold")

ax.axis("equal")

plt.tight_layout()
plt.show()


# Tabla resumen
csp = (
    becario_maestria["ESTADO_DE_CSP"]
    .value_counts()
    .reset_index()
)

csp.columns = ["ESTADO_DE_CSP", "cantidad"]

# Porcentajes
csp["porcentaje"] = (
    csp["cantidad"] /
    csp["cantidad"].sum() * 100
).round(1)

# Etiquetas amigables
csp["estado_mostrar"] = csp["ESTADO_DE_CSP"].replace({
    "CUMPLIO": "Cumplió",
    "PENDIENTE": "Pendiente",
    "SIN INFORMACIÓN": "Sin información",
    "INCUMPLIMIENTO": "Incumplimiento"
})

# Etiquetas con frecuencia
csp["label"] = (
    csp["estado_mostrar"]
    + "\n(n="
    + csp["cantidad"].map("{:,}".format)
    + ")"
)

labels = csp["label"]
sizes = csp["porcentaje"]

# Colores institucionales
colors = [
    "#0B4F6C",  # Azul petróleo
    "#5FB7C6",  # Celeste institucional
    "#A3AD2C",  # Verde oliva
    "#D9D9D9"   # Gris neutro
]

# Mostrar porcentaje solo si es >= 1%
def formato_pct(pct):
    return f"{pct:.1f}%" if pct >= 1 else ""

# Figura
fig, ax = plt.subplots(figsize=(8, 6))

wedges, texts, autotexts = ax.pie(
    sizes,
    labels=labels,
    colors=colors,
    autopct=formato_pct,
    startangle=90,
    counterclock=False,
    wedgeprops={
        "edgecolor": "white",
        "linewidth": 2
    },
    textprops={
        "fontsize": 14,
        "fontweight": "bold"
    }
)

# Estilo porcentajes
for autotext in autotexts:
    autotext.set_color("white")
    autotext.set_fontsize(12)
    autotext.set_fontweight("bold")

ax.axis("equal")

plt.tight_layout()
plt.show()

# Para becarios de programas de doctorado

# Orden fijo de categorías
orden_estados = [
    "CUMPLIO",
    "PENDIENTE",
    "SIN REGISTRO",
    "INCUMPLIMIENTO"
]

# Colores fijos por estado
colores_estado = {
    "CUMPLIO": "#0B4F6C",
    "PENDIENTE": "#5FB7C6",
    "SIN REGISTRO": "#A3AD2C",
    "INCUMPLIMIENTO": "#D9D9D9"
}

# Etiquetas amigables
etiquetas_estado = {
    "CUMPLIO": "Cumplió",
    "PENDIENTE": "Pendiente",
    "SIN REGISTRO": "Sin registro",
    "INCUMPLIMIENTO": "Incumplimiento"
}

# Tabla resumen
csp = (
    becario_doctorado["ESTADO_DE_CSP"]
    .value_counts()
    .reindex(orden_estados, fill_value=0)
    .reset_index()
)

csp.columns = ["ESTADO_DE_CSP", "cantidad"]

# Quitar estados con cero casos
csp = csp[csp["cantidad"] > 0]

# Porcentajes
csp["porcentaje"] = (
    csp["cantidad"] / csp["cantidad"].sum() * 100
).round(1)

# Etiqueta visible
csp["estado_mostrar"] = csp["ESTADO_DE_CSP"].replace(etiquetas_estado)

csp["label"] = (
    csp["estado_mostrar"]
    + "\n(n="
    + csp["cantidad"].map("{:,}".format)
    + ")"
)

labels = csp["label"]
sizes = csp["porcentaje"]

# Colores según estado, no según posición
colors = csp["ESTADO_DE_CSP"].map(colores_estado)

def formato_pct(pct):
    return f"{pct:.1f}%" if pct >= 1 else ""

fig, ax = plt.subplots(figsize=(8, 6))

wedges, texts, autotexts = ax.pie(
    sizes,
    labels=labels,
    colors=colors,
    autopct=formato_pct,
    startangle=90,
    counterclock=False,
    wedgeprops={
        "edgecolor": "white",
        "linewidth": 2
    },
    textprops={
        "fontsize": 14,
        "fontweight": "bold"
    }
)

for autotext in autotexts:
    autotext.set_color("white")
    autotext.set_fontsize(12)
    autotext.set_fontweight("bold")

ax.axis("equal")

plt.tight_layout()
plt.show()

###############################################################################
# Se analiza el estado educativo de los becarios de programas de maestria
###############################################################################
grado_pct = (
    becario_maestria["GRADO_PREGRADO"]
    .value_counts(normalize=True)
    .mul(100)
    .round(1)
    .sort_values(ascending=False)
)

# =====================
# Gráfico
# =====================
fig, ax = plt.subplots(figsize=(12, 8))

bars = ax.bar(
    grado_pct.index,
    grado_pct.values,
    color="#0B4F6C",
    edgecolor="#1F1F1F",
    linewidth=0.8
)

# Etiquetas
for bar, valor in zip(bars, grado_pct.values):
    ax.text(
        bar.get_x() + bar.get_width()/2,
        valor + 0.8,
        f"{valor:.1f}%",
        ha="center",
        va="bottom",
        fontsize=16,
        fontweight="bold"
    )

# Diseño
#ax.set_title(
    #"Distribución porcentual del grado de pregrado",
    #fontsize=16,
    #fontweight="bold",
    #pad=15
#)

ax.set_ylabel("Porcentaje (%)", fontsize=13)
ax.set_xlabel("")
ax.set_ylim(0, grado_pct.max() + 8)

plt.xticks(rotation=20, ha="right", fontsize=14)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.3)

plt.tight_layout()
plt.show()



###############################################################################
# Se analiza el estado educativo de los becarios de programas de doctorado
###############################################################################
grado_pct = (
    becario_doctorado["GRADO_PREGRADO"]
    .value_counts(normalize=True)
    .mul(100)
    .round(1)
    .sort_values(ascending=False)
)

# =====================
# Gráfico
# =====================
fig, ax = plt.subplots(figsize=(12, 8))

bars = ax.bar(
    grado_pct.index,
    grado_pct.values,
    color="#0B4F6C",
    edgecolor="#1F1F1F",
    linewidth=0.8
)

# Etiquetas
for bar, valor in zip(bars, grado_pct.values):
    ax.text(
        bar.get_x() + bar.get_width()/2,
        valor + 0.8,
        f"{valor:.1f}%",
        ha="center",
        va="bottom",
        fontsize=16,
        fontweight="bold"
    )

# Diseño
#ax.set_title(
    #"Distribución porcentual del grado de pregrado",
    #fontsize=16,
    #fontweight="bold",
    #pad=15
#)

ax.set_ylabel("Porcentaje (%)", fontsize=13)
ax.set_xlabel("")
ax.set_ylim(0, grado_pct.max() + 8)

plt.xticks(rotation=20, ha="right", fontsize=14)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.3)

plt.tight_layout()
plt.show()


###############################################################################
# Se analiza el tipo de gestión de la universdad del pregrado de becarios tanto
# de programas de maestria como doctorado
###############################################################################

# Becarios de programas de maestria
gestion1 = (
    becario_maestria["GESTION_PREGRADO"]
    .value_counts()
    .reset_index()
)

gestion1.columns = ["GESTION_PREGRADO", "count"]

# Porcentajes
gestion1["porcentaje"] = (
    gestion1["count"] /
    gestion1["count"].sum() * 100
).round(1)

# Etiquetas amigables
gestion1["estado_mostrar"] = gestion1["GESTION_PREGRADO"].replace({
    "SIN REGISTRO": "Sin registro",
    "PÚBLICO": "Público",
    "PRIVADO": "Privado"
})

# Etiquetas con frecuencia
gestion1["label"] = (
    gestion1["estado_mostrar"]
    + "\n(n="
    + gestion1["count"].map("{:,}".format)
    + ")"
)

labels = gestion1["label"]
sizes = gestion1["porcentaje"]

# Colores institucionales
colors = [
    "#0B4F6C",  # Azul petróleo
    "#5FB7C6",  # Celeste institucional
    "#A3AD2C",  # Verde oliva
    "#D9D9D9"   # Gris neutro
]

# Mostrar porcentaje solo si es >= 1%
def formato_pct(pct):
    return f"{pct:.1f}%" if pct >= 1 else ""

# Figura
fig, ax = plt.subplots(figsize=(8, 6))

wedges, texts, autotexts = ax.pie(
    sizes,
    labels=labels,
    colors=colors,
    autopct=formato_pct,
    startangle=90,
    counterclock=False,
    wedgeprops={
        "edgecolor": "white",
        "linewidth": 2
    },
    textprops={
        "fontsize": 14,
        "fontweight": "bold"
    }
)

# Estilo porcentajes
for autotext in autotexts:
    autotext.set_color("white")
    autotext.set_fontsize(12)
    autotext.set_fontweight("bold")

ax.axis("equal")

plt.tight_layout()
plt.show()


# Becarios de programas de doctorado
gestion2 = (
    becario_doctorado["GESTION_PREGRADO"]
    .value_counts()
    .reset_index()
)

gestion2.columns = ["GESTION_PREGRADO", "count"]

# Porcentajes
gestion2["porcentaje"] = (
    gestion2["count"] /
    gestion2["count"].sum() * 100
).round(1)

# Etiquetas amigables
gestion2["estado_mostrar"] = gestion2["GESTION_PREGRADO"].replace({
    "SIN REGISTRO": "Sin registro",
    "PÚBLICO": "Público",
    "PRIVADO": "Privado"
})

# Etiquetas con frecuencia
gestion2["label"] = (
    gestion2["estado_mostrar"]
    + "\n(n="
    + gestion2["count"].map("{:,}".format)
    + ")"
)

labels = gestion2["label"]
sizes = gestion2["porcentaje"]

# Colores institucionales
colors = [
    "#0B4F6C",  # Azul petróleo
    "#5FB7C6",  # Celeste institucional
    "#A3AD2C",  # Verde oliva
    "#D9D9D9"   # Gris neutro
]

# Mostrar porcentaje solo si es >= 1%
def formato_pct(pct):
    return f"{pct:.1f}%" if pct >= 1 else ""

# Figura
fig, ax = plt.subplots(figsize=(8, 6))

wedges, texts, autotexts = ax.pie(
    sizes,
    labels=labels,
    colors=colors,
    autopct=formato_pct,
    startangle=90,
    counterclock=False,
    wedgeprops={
        "edgecolor": "white",
        "linewidth": 2
    },
    textprops={
        "fontsize": 14,
        "fontweight": "bold"
    }
)

# Estilo porcentajes
for autotext in autotexts:
    autotext.set_color("white")
    autotext.set_fontsize(12)
    autotext.set_fontweight("bold")

ax.axis("equal")

plt.tight_layout()
plt.show()


###############################################################################
# Se analiza la universidad de procedencia (Top 10) de los becarios de programa
# de maestria y doctorado
###############################################################################

# Se realiza el análisis para becarios de programas de maestría
total = becario_maestria["INSTITUCION_ORIGEN_PREGRADO"].value_counts()

proporcion = (
    becario_maestria["INSTITUCION_ORIGEN_PREGRADO"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

pais_segunda_maestria = pd.DataFrame({
    "TOTAL": total,
    "PROPORCION (%)": proporcion
})

pais_segunda_maestria
pais_segunda_maestria.reset_index(inplace=True)
pais_segunda_maestria = pais_segunda_maestria.head(10)
pais_segunda_maestria.columns

pais_segunda_maestria = pais_segunda_maestria.sort_values("TOTAL", ascending=True)

color_principal = "#0B4F6C"   # azul institucional
color_secundario = "#5FB7C6"  # celeste

plt.figure(figsize=(12, 6))

bars = plt.barh(
    pais_segunda_maestria["INSTITUCION_ORIGEN_PREGRADO"],
    pais_segunda_maestria["PROPORCION (%)"],
    color=color_principal
)

# Etiquetas: porcentaje + total
for i, (pct, total) in enumerate(
    zip(
        pais_segunda_maestria["PROPORCION (%)"],
        pais_segunda_maestria["TOTAL"]
    )
):
    plt.text(
        pct + 0.5,
        i,
        f"{pct:.1f}% ({total:,})",
        va="center",
        fontsize=18
    )

plt.xlabel("Participación (%)")

plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

plt.grid(axis="x", linestyle="--", alpha=0.3)

plt.tight_layout()
plt.show()


# Se realiza el análisis para becarios de programas de maestría
total = becario_doctorado["INSTITUCION_ORIGEN_PREGRADO"].value_counts()

proporcion = (
    becario_doctorado["INSTITUCION_ORIGEN_PREGRADO"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

pais_segunda_maestria = pd.DataFrame({
    "TOTAL": total,
    "PROPORCION (%)": proporcion
})

pais_segunda_maestria
pais_segunda_maestria.reset_index(inplace=True)
pais_segunda_maestria = pais_segunda_maestria.head(10)
pais_segunda_maestria.columns

pais_segunda_maestria = pais_segunda_maestria.sort_values("TOTAL", ascending=True)

color_principal = "#0B4F6C"   # azul institucional
color_secundario = "#5FB7C6"  # celeste

plt.figure(figsize=(12, 6))

bars = plt.barh(
    pais_segunda_maestria["INSTITUCION_ORIGEN_PREGRADO"],
    pais_segunda_maestria["PROPORCION (%)"],
    color=color_principal
)

# Etiquetas: porcentaje + total
for i, (pct, total) in enumerate(
    zip(
        pais_segunda_maestria["PROPORCION (%)"],
        pais_segunda_maestria["TOTAL"]
    )
):
    plt.text(
        pct + 0.5,
        i,
        f"{pct:.1f}% ({total:,})",
        va="center",
        fontsize=18
    )

plt.xlabel("Participación (%)")

plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

plt.grid(axis="x", linestyle="--", alpha=0.3)

plt.tight_layout()
plt.show()

###############################################################################
# Se analiza la distribución de los RENACYT de los becarios de programas de
# maestría
###############################################################################

# Becarios de programas de maestria
renacyt_maestria = (
    becario_maestria["RENACYT"]
    .value_counts()
    .reset_index()
)

renacyt_maestria.columns = ["RENACYT", "count"]

# Porcentajes
renacyt_maestria["porcentaje"] = (
    renacyt_maestria["count"] /
    renacyt_maestria["count"].sum() * 100
).round(1)

# Etiquetas amigables
renacyt_maestria["estado_mostrar"] = renacyt_maestria["RENACYT"].replace({
    "SIN REGISTRO EN RENACYT": "SIN REGISTRO EN RENACYT",
    "REGISTRADO EN RENACYT": "REGISTRADO EN RENACYT"
})

# Etiquetas con frecuencia
renacyt_maestria["label"] = (
    renacyt_maestria["estado_mostrar"]
    + "\n(n="
    + renacyt_maestria["count"].map("{:,}".format)
    + ")"
)

labels = renacyt_maestria["label"]
sizes = renacyt_maestria["porcentaje"]

# Colores institucionales
colors = [
    "#0B4F6C",  # Azul petróleo
    "#5FB7C6",  # Celeste institucional
    "#A3AD2C",  # Verde oliva
    "#D9D9D9"   # Gris neutro
]

# Mostrar porcentaje solo si es >= 1%
def formato_pct(pct):
    return f"{pct:.1f}%" if pct >= 1 else ""

# Figura
fig, ax = plt.subplots(figsize=(8, 6))

wedges, texts, autotexts = ax.pie(
    sizes,
    labels=labels,
    colors=colors,
    autopct=formato_pct,
    startangle=90,
    counterclock=False,
    wedgeprops={
        "edgecolor": "white",
        "linewidth": 2
    },
    textprops={
        "fontsize": 14,
        "fontweight": "bold"
    }
)

# Estilo porcentajes
for autotext in autotexts:
    autotext.set_color("white")
    autotext.set_fontsize(12)
    autotext.set_fontweight("bold")

ax.axis("equal")

plt.tight_layout()
plt.show()

# Becarios de programas de doctorado
renacyt_doctorado = (
    becario_doctorado["RENACYT"]
    .value_counts()
    .reset_index()
)

renacyt_doctorado.columns = ["RENACYT", "count"]

# Porcentajes
renacyt_doctorado["porcentaje"] = (
    renacyt_doctorado["count"] /
    renacyt_doctorado["count"].sum() * 100
).round(1)

# Etiquetas amigables
renacyt_doctorado["estado_mostrar"] = renacyt_doctorado["RENACYT"].replace({
    "SIN REGISTRO EN RENACYT": "SIN REGISTRO EN RENACYT",
    "REGISTRADO EN RENACYT": "REGISTRADO EN RENACYT"
})

# Etiquetas con frecuencia
renacyt_doctorado["label"] = (
    renacyt_doctorado["estado_mostrar"]
    + "\n(n="
    + renacyt_doctorado["count"].map("{:,}".format)
    + ")"
)

labels = renacyt_doctorado["label"]
sizes = renacyt_doctorado["porcentaje"]

# Colores institucionales
colors = [
    "#0B4F6C",  # Azul petróleo
    "#5FB7C6",  # Celeste institucional
    "#A3AD2C",  # Verde oliva
    "#D9D9D9"   # Gris neutro
]

# Mostrar porcentaje solo si es >= 1%
def formato_pct(pct):
    return f"{pct:.1f}%" if pct >= 1 else ""

fig, ax = plt.subplots(figsize=(8, 6))

wedges, _, autotexts = ax.pie(
    sizes,
    labels=None,
    colors=colors,
    autopct=formato_pct,
    startangle=90,
    counterclock=False,
    radius=1.20,
    pctdistance=0.62,
    wedgeprops={
        "edgecolor": "white",
        "linewidth": 2
    }
)

# Porcentajes internos
for autotext in autotexts:
    autotext.set_color("white")
    autotext.set_fontsize(14)
    autotext.set_fontweight("bold")

# Etiqueta superior izquierda
ax.text(
    -0.60, 1.18,
    labels.iloc[1],
    ha="right",
    va="bottom",
    fontsize=14,
    fontweight="bold"
)

# Etiqueta inferior derecha
ax.text(
    0.45, -1.12,
    labels.iloc[0],
    ha="left",
    va="top",
    fontsize=14,
    fontweight="bold"
)

# Límites ajustados al pastel
ax.set_xlim(-1.55, 1.55)
ax.set_ylim(-1.40, 1.40)

ax.set_aspect("equal")
ax.axis("off")

plt.subplots_adjust(
    left=0.02,
    right=0.98,
    top=0.98,
    bottom=0.02
)

plt.show()

###############################################################################
# Se analiza las regiones de los becarios de programas de maestria y doctorado
###############################################################################

# Becarios de programas de maestria

peru_maestria = becario_maestria.REGION_PREGRADO.value_counts()
peru_maestria = peru_maestria.to_frame()
peru_maestria.reset_index(inplace=True)
peru_maestria.rename(columns=({"REGION_PREGRADO":"Region", "count":"cantidad"}), inplace=True)


#código para la elaboración de un mapa de calor, considerando una variable específica, utilizando geopandas
# Se construye el mapa de Perú, a nivel departamental
peru = geopandas.read_file("INEI_LIMITE_DEPARTAMENTAL_GEOGPSPERU_JUANSUYO_931381206.shp")
peru.columns
peru.rename(columns=({"NOMBDEP":"Region"}), inplace=True)

# Se fusiona el dataframe peru con region
peru = pd.merge(peru,peru_maestria, on="Region", how="left")


#Paleta institucional
custom_cmap = LinearSegmentedColormap.from_list(
    "prociencia_palette",
    [
        "#E6F4F7",  # celeste muy claro
        "#00A7B5",  # turquesa
        "#1F6F8B"   # azul petróleo
    ]
)

fig, ax = plt.subplots(figsize=(10, 6))

ax.axis("off")

# Mapa base (regiones sin datos en gris)
peru.plot(
    ax=ax,
    color="#D9D9D9",
    edgecolor="white",
    linewidth=0.8
)

# Regiones con datos
peru.plot(
    column="cantidad",
    cmap=custom_cmap,
    linewidth=0.8,
    ax=ax,
    edgecolor="white",
    legend=True
)

# Etiquetas
for idx, row in peru.iterrows():

    if pd.notna(row["cantidad"]):

        # Coordenadas del centroide
        x = row.geometry.centroid.x
        y = row.geometry.centroid.y

        # Texto blanco para regiones oscuras
        color_texto = "white" if row["cantidad"] >= 20 else "black"

        ax.annotate(
            text=f"{int(row['cantidad'])}",
            xy=(x, y),
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=color_texto
        )

plt.tight_layout()
plt.show()



###############################################################################
# Modelo de regresión logística
###############################################################################

# Se construye un modelo de regresión logística
# utilizando el dataframe becarios, que representa el merge entre stem y pronabec_becario
becarios.shape
becarios.columns
becarios.info()

# Se cuenta la cantidad de registros que tiene el campo STEM del dataframe fusion
becarios["STEM"].count()


# Se analiza la distribución de la condición STEM de los becarios tanto de programas de maestría como doctorado
becarios.STEM.value_counts()
becarios.STEM.value_counts(normalize=True).round(2)*100

# Se analiza la distribución del campo ESTADO_DE_CSP
becarios["ESTADO_DE_CSP"].count()
becarios.ESTADO_DE_CSP.value_counts()
becarios.ESTADO_DE_CSP.value_counts(normalize=True).round(3)*100

# Se crea explicitamente la categoría "NO CUMPLIÓ", y se construye una variable booleana
becarios["CUMPLIMIENTO_CAT"] = becarios["ESTADO_DE_CSP"].replace(
    {
        "PENDIENTE": "NO CUMPLIO",
        "SIN REGISTRO": "NO CUMPLIO",
        "INCUMPLIMIENTO": "NO CUMPLIO"
    }
)


becarios["CUMPLIO_BIN"] = becarios["CUMPLIMIENTO_CAT"].map(
    {
        "CUMPLIO": 1,
        "NO CUMPLIO": 0
    }
)

becarios.columns

# En el caso de Pais de destino se agrupan por cantidad de becarios
top_paises = (
    becarios["PAISDESTINO"]
    .value_counts()
    .head(10)
    .index
)

becarios["PAISDESTINO_GRP"] = np.where(
    becarios["PAISDESTINO"].isin(top_paises),
    becarios["PAISDESTINO"],
    "OTROS"
)


# Dado que mi objetivo es estimar la probabilidad de cumplimiento de becarios en el extranjero
# entonces tengo que deliminar la poblicación de mi estudio
becarios = becarios[
    becarios["PAISDESTINO_GRP"] != "PERU"
].copy()


# Se identifica si el dataframe tiene algunos valores NaN
becarios[becarios.isna().any(axis=1)]
becarios.isna().sum()

###############################################################################
# Se realiza un análisis univariado
###############################################################################
becarios["CUMPLIO_BIN"].value_counts(normalize=True)
becarios["STEM"].value_counts(normalize=True)
becarios["NIVEL_EDUCATIVO"].value_counts(normalize=True)


###############################################################################
# Se realiza un análisis bivariado
###############################################################################
pd.crosstab(
    becarios["STEM"],
    becarios["CUMPLIO_BIN"],
    normalize="index"
)*100


pd.crosstab(
    becarios["NIVEL_EDUCATIVO"],
    becarios["CUMPLIO_BIN"],
    normalize="index"
)*100


pd.crosstab(
    becarios["STEM"],
    becarios["NIVEL_EDUCATIVO"],
    normalize="index"
)


pd.crosstab(
    becarios["STEM"],
    becarios["RENACYT"],
    normalize="index"
)*100



pd.crosstab(
    [becarios["NIVEL_EDUCATIVO"],
     becarios["STEM"]],
    becarios["CUMPLIO_BIN"],
    normalize="index"
) * 100


pd.crosstab(
    [becarios["STEM"],
     becarios["RENACYT"]],
    becarios["CUMPLIO_BIN"],
    normalize="index"
) * 100



# Se construye el dataset que permitirá hacer el análisis
modelo = becarios[["CUMPLIO_BIN", "EDADBASES", "SEXO", "STEM", "NIVEL_EDUCATIVO", "PAISDESTINO_GRP"]]

# Seleccionar variables

df_model = modelo[
    [
        "CUMPLIO_BIN",
        "EDADBASES",
        "SEXO",
        "STEM",
        "NIVEL_EDUCATIVO",
        "PAISDESTINO_GRP"
    ]
].copy()


df_model = df_model.dropna()

# Se revisa la variable PAISDESTINO_GRP de mi dataframe df_model
df_model["PAISDESTINO_GRP"].value_counts()


df_model["CUMPLIO_BIN"] = df_model["CUMPLIO_BIN"].astype(int)
df_model["EDADBASES"] = pd.to_numeric(df_model["EDADBASES"], errors="coerce")

# Se elabora un gráfico pie para observar la distribución del cumplimiento del servicio
# con el país
csp = (
    df_model["CUMPLIO_BIN"]
    .value_counts()
    .reset_index()
)

csp.columns = ["ESTADO_DE_CSP", "cantidad"]

# Porcentajes
csp["porcentaje"] = (
    csp["cantidad"] /
    csp["cantidad"].sum() * 100
).round(1)

# Etiquetas amigables
csp["estado_mostrar"] = csp["ESTADO_DE_CSP"].replace({
    1: "CUMPLIÓ",
    0: "NO CUMPLIÓ"
})

# Etiquetas con frecuencia
csp["label"] = (
    csp["estado_mostrar"]
    + "\n(n="
    + csp["cantidad"].map("{:,}".format)
    + ")"
)

labels = csp["label"]
sizes = csp["porcentaje"]

# Colores institucionales
colors = [
    "#0B4F6C",  # Azul petróleo
    "#5FB7C6",  # Celeste institucional
]

# Mostrar porcentaje solo si es >= 1%
def formato_pct(pct):
    return f"{pct:.1f}%" if pct >= 1 else ""

# Figura
fig, ax = plt.subplots(figsize=(8, 6))

wedges, texts, autotexts = ax.pie(
    sizes,
    labels=labels,
    colors=colors,
    autopct=formato_pct,
    startangle=90,
    counterclock=False,
    wedgeprops={
        "edgecolor": "white",
        "linewidth": 2
    },
    textprops={
        "fontsize": 16,
        "fontweight": "bold"
    }
)

# Estilo porcentajes
for autotext in autotexts:
    autotext.set_color("white")
    autotext.set_fontsize(18)
    autotext.set_fontweight("bold")

ax.axis("equal")

plt.tight_layout()
plt.show()


formula = """
CUMPLIO_BIN ~ EDADBASES
            + C(SEXO)
            + C(STEM)
            + C(NIVEL_EDUCATIVO)
            + C(PAISDESTINO_GRP, Treatment(reference='ESPAÑA'))
"""

logit_model = smf.logit(formula=formula, data=df_model).fit()

print(logit_model.summary())


params = logit_model.params
conf = logit_model.conf_int()
pvalues = logit_model.pvalues

odds_ratios = pd.DataFrame({
    "coeficiente": params,
    "odds_ratio": np.exp(params),
    "ic_95_inf": np.exp(conf[0]),
    "ic_95_sup": np.exp(conf[1]),
    "p_value": pvalues
})

odds_ratios = odds_ratios.sort_values("p_value")

odds_ratios

odds_ratios[odds_ratios["p_value"] < 0.05]


from patsy import dmatrices

y, X = dmatrices(
    formula,
    data=df_model,
    return_type="dataframe"
)

X.head()


from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()

vif["Variable"] = X.columns

vif["VIF"] = [
    variance_inflation_factor(X.values, i)
    for i in range(X.shape[1])
]

vif.sort_values("VIF", ascending=False)

