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









































x = range(len(pronabec_postu))

color_no = "#A3AD2C"
color_si = "#5FB7C6"
color_total = "#0B4F6C"

fig, ax = plt.subplots(figsize=(13, 6))

# =====================
# Barras apiladas
# No se adjudicó abajo, Se adjudicó arriba
# =====================
b_maestria = ax.bar(
    x,
    pronabec_postu["NO SE LE ADJUDICÓ LA BECA"],
    color=color_no,
    width=0.72,
    label="No se adjudicó"
)

b_doctorado = ax.bar(
    x,
    pronabec_postu["SE LE ADJUDICÓ LA BECA"],
    bottom=pronabec_postu["NO SE LE ADJUDICÓ LA BECA"],
    color=color_si,
    width=0.72,
    label="Se adjudicó"
)

# =====================
# Etiquetas Maestría
# =====================
for i, mae in enumerate(pronabec_postu["NO SE LE ADJUDICÓ LA BECA"]):
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
for i, (mae, doc) in enumerate(zip(pronabec_postu["NO SE LE ADJUDICÓ LA BECA"], pronabec_postu["SE LE ADJUDICÓ LA BECA"])):
    if doc > 0:
        ax.text(
            i,
            mae + doc + 18,
            f"{doc:,.0f}",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
            color=color_si
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
        fontsize=13,
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














