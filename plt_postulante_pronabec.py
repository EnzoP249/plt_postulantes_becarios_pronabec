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
from matplotlib.ticker import FuncFormatter
from rapidfuzz import process, fuzz
from unidecode import unidecode
import geopandas
from shapely.geometry import Point, Polygon
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mticker
import statsmodels.api as sm
import statsmodels.formula.api as smf

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



# Se carga el archivo en formato xlsx denominado 0_BGB_2013_2025, el cual
# contiene información de los becarios, y se almacena en un objeto dataframe

# Se construye una función que aborde la conversión de int en str para un procesamiento óptimizado
def int_to_str(value):
    return str(value)

# Especifica el diccionario de conversión en el parámetro converters
converters = {"ID_POSTULACION": int_to_str}

pronabec = pd.read_excel("BGB_2013_2025_updt.xlsx", sheet_name="Sheet1", header=0, converters=converters)

# Se identifican caracteristicas estructurales del dataframe pronabec
pronabec.shape
pronabec.info()
pronabec.dtypes
pronabec.columns


# Se reorganiza la información que integra el objeto dataframe pronabec
pronabec = pronabec[["N","ID_POSTULACION", "CONDICION_FINAL", "NEXPEDIENTE", "AÑO_CONVOCATORIA","SEXO", "EDADBASES",
                     "PAISDESTINO", "NIVEL_EDUCATIVO","INSTITUCION", "TIPO_GESTION", "NOMBRECARRERA", "AREA_CONOCIMIENTO",
                            "FECHA_INICIO_PROGRAMA", "FECHA_FIN_PROGRAMA", "DURACION_DIAS",
                            'COMPROMISO_DE_SERVICIO_AL_PAIS', 'ESTADO_DE_CSP',
                            'TIEMPO_MAXIMO_COMPROMISO', 'AÑO_ACREDITACION', 'EXPERGENERALMESES',
                            'GRADO_PREGRADO', 'INSTITUCION_ORIGEN_PREGRADO', 'GESTION_PREGRADO',
                            'ESPECIALIDAD_PREGRADO', 'REGION_PREGRADO']]


# Se analiza cada atributo o campo del dataframe pronabec
pronabec.ID_POSTULACION.value_counts()
pronabec.CONDICION_FINAL.value_counts()
pronabec.PAISDESTINO.value_counts()
pronabec.NIVEL_EDUCATIVO.value_counts()
pronabec.INSTITUCION.value_counts()
pronabec.TIPO_GESTION.value_counts()


# Se cuentan los identificadores de postulaciones únicos
pronabec["ID_POSTULACION"].nunique()

# En total, durante el periodo de análisis, ¿cuántos postulantes no recibieron la beca?
pronabec_no = pronabec[pronabec["CONDICION_FINAL"]=="NO SE LE ADJUDICÓ LA BECA"]

# Se cuentan los identificadores de postulaciones únicas considerando el dataframe pronabec_no
pronabec_no["ID_POSTULACION"].nunique()

###############################################################################
# Se realiza un análisis de los postulantes a la Beca Generación del Bicentenario
# El análisis considera a los postulantes que tienen código de postulación
###############################################################################

# Se eliminan los valores nan del dataframe pronabec
postulante = pronabec.dropna(subset=["ID_POSTULACION"])

# Se identifican las categorias del campo nivel educativo
postulante.NIVEL_EDUCATIVO.value_counts()
postulante = postulante[postulante["NIVEL_EDUCATIVO"]!="NO CODIFICADA"]

# Se identifica si existen valores duplicados en el dataframe postulante
postulante["ID_POSTULACION"].nunique()

# Se analiza la distribución de postulantes por genero
postulante.SEXO.value_counts(normalize=True).round(2)*100

###############################################################################
# Se realiza un análisis de la edad de los postulantes
###############################################################################
promedio_edad = postulante.groupby("SEXO")["EDADBASES"].mean()
print(promedio_edad)

edad_resumen = (
    postulante.groupby("AÑO_CONVOCATORIA")["EDADBASES"]
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
fig, ax = plt.subplots(figsize=(12, 6))

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
        xytext=(-12, -8),
        textcoords="offset points",
        fontsize=8,
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
        fontsize=10,
        color="#0B4F6C",
        fontweight="bold"
    )

    # Edad máxima
    ax.annotate(
        f'{row["EDAD_MAXIMA"]:.0f}',
        (row["AÑO_CONVOCATORIA"], row["EDAD_MAXIMA"]),
        xytext=(10, 6),
        textcoords="offset points",
        fontsize=8,
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
    bbox_to_anchor=(0.5, 1.08)
)

plt.tight_layout()
plt.show()


# Se obtiene un grafico juntos para el nivel educativo del postulante
postulante_edu = pd.pivot_table(postulante, values="ID_POSTULACION", index="AÑO_CONVOCATORIA", columns="NIVEL_EDUCATIVO", aggfunc="count")
postulante_edu.reset_index(inplace=True)

# Se reemplazan algunas columnas con valor nan a 0
postulante_edu["DOCTORADO"] = postulante_edu["DOCTORADO"].fillna(0)

# Se calcula un campo que representa la suma entre los niveles educativos
postulante_edu["TOTAL"] = postulante_edu["DOCTORADO"] + postulante_edu["MAESTRIA"]

# Se convierte año convocatoria a una variable string
postulante_edu["AÑO_CONVOCATORIA"] = postulante_edu["AÑO_CONVOCATORIA"].astype(str)


x = range(len(postulante_edu))

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
    postulante_edu["MAESTRIA"],
    color=color_maestria,
    width=0.72,
    label="Maestría"
)

b_doctorado = ax.bar(
    x,
    postulante_edu["DOCTORADO"],
    bottom=postulante_edu["MAESTRIA"],
    color=color_doctorado,
    width=0.72,
    label="Doctorado"
)

# =====================
# Etiquetas Maestría
# =====================
for i, mae in enumerate(postulante_edu["MAESTRIA"]):
    if mae >= 80:
        ax.text(
            i,
            mae / 2,
            f"{mae:,.0f}",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="white"
        )

# =====================
# Etiquetas Doctorado
# Se colocan encima del segmento para que no se pierdan
# =====================
for i, (mae, doc) in enumerate(zip(postulante_edu["MAESTRIA"], postulante_edu["DOCTORADO"])):
    if doc > 0:
        ax.text(
            i,
            mae + doc + 18,
            f"{doc:,.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color=color_doctorado
        )

# =====================
# Etiquetas Total
# =====================
for i, total in enumerate(postulante_edu["TOTAL"]):
    ax.text(
        i,
        total + max(postulante_edu["TOTAL"]) * 0.040,
        f"{total:,.0f}",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color=color_total
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
ax.set_xticklabels(postulante_edu["AÑO_CONVOCATORIA"])

ax.yaxis.set_major_formatter(
    mticker.StrMethodFormatter("{x:,.0f}")
)

ax.set_ylim(0, postulante_edu["TOTAL"].max() * 1.15)

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


# Se calcula la distribución de postulantes por año
postulante_año = pd.pivot_table(postulante, values="ID_POSTULACION", index="AÑO_CONVOCATORIA", aggfunc="count")
postulante_año.reset_index(inplace=True)

# Se transforman campos del dataframe pronabec_becario_año
postulante_año["AÑO_CONVOCATORIA"] = pd.to_numeric(postulante_año["AÑO_CONVOCATORIA"], errors="coerce")
postulante_año["ID_POSTULACION"] = pd.to_numeric(postulante_año["ID_POSTULACION"], errors="coerce")

# Se renombran atributos del dataframe pronabec_becario_año
postulante_año.rename(columns=({"ID_POSTULACION":"CANTIDAD"}), inplace=True)

# Se organiza en función de año
postulante_año = postulante_año.sort_values("AÑO_CONVOCATORIA")

fig, ax = plt.subplots(figsize=(10, 6))

barras = ax.bar(
    postulante_año["AÑO_CONVOCATORIA"],
    postulante_año["CANTIDAD"],
    color="#0B4F6C",
    edgecolor="white",
    linewidth=1.5
)

# Etiquetas arriba de cada columna
for barra in barras:
    alto = barra.get_height()
    ax.text(
        barra.get_x() + barra.get_width() / 2,
        alto,
        f"{alto:,.0f}",
        ha="center",
        va="bottom",
        fontsize=10,
        #fontweight="bold",
        color="black"
    )

# Mostrar todos los años
ax.set_xticks(postulante_año["AÑO_CONVOCATORIA"])
ax.set_xticklabels(
    postulante_año["AÑO_CONVOCATORIA"],
    rotation=0
)

#ax.set_title(
    #"Evolución del número total de postulantes, 2013-2025",
    #fontsize=14,
    #fontweight="bold"
#)

plt.xlabel("Año")
plt.ylabel("Número de postulantes")


ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()

###############################################################################
# Se analiza los postulantes a programas de maestria
###############################################################################
postulante_maestria = postulante[postulante["NIVEL_EDUCATIVO"]=="MAESTRIA"]

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


fig, ax = plt.subplots(figsize=(10, 6))

# Línea
ax.plot(
    postulante_maestria_año["AÑO_CONVOCATORIA"],
    postulante_maestria_año["CANTIDAD"],
    color="red",
    marker="o",
    linewidth=2.5,
    markersize=8
)

# Etiquetas
for x, y in zip(postulante_maestria_año["AÑO_CONVOCATORIA"], postulante_maestria_año["CANTIDAD"]):
    ax.annotate(
        f"{y:,}",
        (x, y),
        textcoords="offset points",
        xytext=(0, 8),
        ha="center",
        fontsize=10,
        #fontweight="bold"
    )

# Formato
#ax.set_title(
    #"Número de becarios por año de convocatoria",
    #fontsize=14,
    #fontweight="bold"
#)

ax.set_xlabel("Año")
ax.set_ylabel("Número de postulantes a programas de maestría")

ax.set_xticks(postulante_maestria_año["AÑO_CONVOCATORIA"])

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)

plt.tight_layout()
plt.show()

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
fig, ax = plt.subplots(figsize=(12, 6))

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
        xytext=(-12, -8),
        textcoords="offset points",
        fontsize=10,
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
        fontsize=10,
        color="#0B4F6C",
        fontweight="bold"
    )

    # Edad máxima
    ax.annotate(
        f'{row["EDAD_MAXIMA"]:.0f}',
        (row["AÑO_CONVOCATORIA"], row["EDAD_MAXIMA"]),
        xytext=(10, 6),
        textcoords="offset points",
        fontsize=10,
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
    bbox_to_anchor=(0.5, 1.08)
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
        fontsize=11
    )

    ax.text(
        año,
        fem + masc/2,
        f"{masc:.1f}%",
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
        fontsize=11
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


###############################################################################
# Se analiza los postulantes a programas de doctorado
###############################################################################
postulante_doctorado = postulante[postulante["NIVEL_EDUCATIVO"]=="DOCTORADO"]

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


fig, ax = plt.subplots(figsize=(10, 6))

# Línea
ax.plot(
    postulante_doctorado_año["AÑO_CONVOCATORIA"],
    postulante_doctorado_año["CANTIDAD"],
    color="purple",
    marker="o",
    linewidth=2.5,
    markersize=8
)

# Etiquetas
for x, y in zip(postulante_doctorado_año["AÑO_CONVOCATORIA"], postulante_doctorado_año["CANTIDAD"]):
    ax.annotate(
        f"{y:,}",
        (x, y),
        textcoords="offset points",
        xytext=(0, 8),
        ha="center",
        fontsize=10,
        #fontweight="bold"
    )

# Formato
#ax.set_title(
    #"Número de becarios por año de convocatoria",
    #fontsize=14,
    #fontweight="bold"
#)

ax.set_xlabel("Año")
ax.set_ylabel("Número de postulantes a programas de doctorado")

ax.set_xticks(postulante_maestria_año["AÑO_CONVOCATORIA"])

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)

plt.tight_layout()
plt.show()


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
fig, ax = plt.subplots(figsize=(12, 6))

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
        xytext=(-12, -8),
        textcoords="offset points",
        fontsize=10,
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
        fontsize=10,
        color="#0B4F6C",
        fontweight="bold"
    )

    # Edad máxima
    ax.annotate(
        f'{row["EDAD_MAXIMA"]:.0f}',
        (row["AÑO_CONVOCATORIA"], row["EDAD_MAXIMA"]),
        xytext=(10, 6),
        textcoords="offset points",
        fontsize=10,
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
    bbox_to_anchor=(0.5, 1.08)
)

plt.tight_layout()
plt.show()


# Se calcula el sexo de los postulantes de doctorado
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
        fontsize=11
    )

    ax.text(
        año,
        fem + masc/2,
        f"{masc:.1f}%",
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
        fontsize=11
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


###############################################################################
# Se realiza un análisis de becarios en función del dataframe postulante
###############################################################################

# En total, durante el periodo de análisis, ¿cuántos postulantes recibieron la beca?
pronabec_becario = postulante[postulante["CONDICION_FINAL"]=="SE LE ADJUDICÓ LA BECA"]
a = pronabec_becario["ID_POSTULACION"].count()
print(f"Los postulantes que recibieron la beca, durante el periodo de análisis, fueron {a}, lo que representa el 19%")

# Se cuentan los identificadores de postulaciones únicas considerando el dataframe pronabec_becario
pronabec_becario["ID_POSTULACION"].nunique()
pronabec_becario.columns

# Se analiza la distribución de postulantes por genero
pronabec_becario.SEXO.value_counts(normalize=True).round(2)*100

# Se calcula la edad promedio para la categoria género
promedio_edad = pronabec_becario.groupby("SEXO")["EDADBASES"].mean()
print(promedio_edad)


edad_resumen = (
    pronabec_becario.groupby("AÑO_CONVOCATORIA")["EDADBASES"]
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
fig, ax = plt.subplots(figsize=(12, 6))

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
        xytext=(-12, -8),
        textcoords="offset points",
        fontsize=10,
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
        fontsize=10,
        color="#0B4F6C",
        fontweight="bold"
    )

    # Edad máxima
    ax.annotate(
        f'{row["EDAD_MAXIMA"]:.0f}',
        (row["AÑO_CONVOCATORIA"], row["EDAD_MAXIMA"]),
        xytext=(10, 6),
        textcoords="offset points",
        fontsize=10,
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
    bbox_to_anchor=(0.5, 1.08)
)

plt.tight_layout()
plt.show()


becario_edu = pd.pivot_table(pronabec_becario, values="ID_POSTULACION", index="AÑO_CONVOCATORIA", columns="NIVEL_EDUCATIVO", aggfunc="count")
becario_edu.reset_index(inplace=True)

# Se reemplazan algunas columnas con valor nan a 0
becario_edu["DOCTORADO"] = becario_edu["DOCTORADO"].fillna(0)

# Se calcula un campo que representa la suma entre los niveles educativos
becario_edu["TOTAL"] = becario_edu["DOCTORADO"] + becario_edu["MAESTRIA"]

# Se convierte año convocatoria a una variable string
becario_edu["AÑO_CONVOCATORIA"] = becario_edu["AÑO_CONVOCATORIA"].astype(str)

x = range(len(becario_edu))

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
    becario_edu["MAESTRIA"],
    color=color_maestria,
    width=0.72,
    label="Maestría"
)

b_doctorado = ax.bar(
    x,
    becario_edu["DOCTORADO"],
    bottom=becario_edu["MAESTRIA"],
    color=color_doctorado,
    width=0.72,
    label="Doctorado"
)

# =====================
# Etiquetas Maestría
# =====================
for i, mae in enumerate(becario_edu["MAESTRIA"]):
    if mae >= 80:
        ax.text(
            i,
            mae / 2,
            f"{mae:,.0f}",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="white"
        )

# =====================
# Etiquetas Doctorado
# Se colocan encima del segmento para que no se pierdan
# =====================
for i, (mae, doc) in enumerate(zip(becario_edu["MAESTRIA"], becario_edu["DOCTORADO"])):
    if doc > 0:
        ax.text(
            i,
            mae + doc + 18,
            f"{doc:,.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color=color_doctorado
        )

# =====================
# Etiquetas Total
# =====================
for i, total in enumerate(becario_edu["TOTAL"]):
    ax.text(
        i,
        total + max(becario_edu["TOTAL"]) * 0.050,
        f"{total:,.0f}",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color=color_total
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
ax.set_xticklabels(becario_edu["AÑO_CONVOCATORIA"])

ax.yaxis.set_major_formatter(
    mticker.StrMethodFormatter("{x:,.0f}")
)

ax.set_ylim(0, becario_edu["TOTAL"].max() * 1.15)

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


# Se calcula la distribución de las categorias que integran la columna ESTADO_DE_CSP
pronabec_becario.ESTADO_DE_CSP.value_counts()

# Renombro valores de la columna ESTADO_DE_CSP
pronabec_becario["ESTADO_DE_CSP"] = pronabec_becario["ESTADO_DE_CSP"].fillna("SIN INFORMACIÓN")

###############################################################################
# Considerando el dataframe pronabec_becario, se obtiene la evolución de los becarios, a nivel temporal
###############################################################################

# Se calcula la distribución de becarios por año
becario_año = pd.pivot_table(pronabec_becario, values="ID_POSTULACION", index="AÑO_CONVOCATORIA", aggfunc="count")
becario_año.reset_index(inplace=True)

# Se transforman campos del dataframe pronabec_becario_año
becario_año["AÑO_CONVOCATORIA"] = pd.to_numeric(becario_año["AÑO_CONVOCATORIA"], errors="coerce")
becario_año["ID_POSTULACION"] = pd.to_numeric(becario_año["ID_POSTULACION"], errors="coerce")

# Se renombran atributos del dataframe becario_año
becario_año.rename(columns=({"ID_POSTULACION":"CANTIDAD"}), inplace=True)

# Se organiza en función de año
becario_año = becario_año.sort_values("AÑO_CONVOCATORIA")

fig, ax = plt.subplots(figsize=(10, 6))

barras = ax.bar(
    becario_año["AÑO_CONVOCATORIA"],
    becario_año["CANTIDAD"],
    color="#2E86AB",
    edgecolor="white",
    linewidth=1.5
)

# Etiquetas arriba de cada columna
for barra in barras:
    alto = barra.get_height()
    ax.text(
        barra.get_x() + barra.get_width() / 2,
        alto,
        f"{alto:,.0f}",
        ha="center",
        va="bottom",
        fontsize=10,
        #fontweight="bold",
        color="black"
    )

# Mostrar todos los años
ax.set_xticks(becario_año["AÑO_CONVOCATORIA"])
ax.set_xticklabels(
    becario_año["AÑO_CONVOCATORIA"],
    rotation=0
)

#ax.set_title(
    #"Evolución del número total de postulantes, 2013-2025",
    #fontsize=14,
    #fontweight="bold"
#)

plt.xlabel("Año")
plt.ylabel("Número de becarios")


ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()


###############################################################################
# Se analiza los becarios de programas de maestria
###############################################################################
becario_maestria = pronabec_becario[pronabec_becario["NIVEL_EDUCATIVO"]=="MAESTRIA"]
becario_maestria.shape

# Se analiza la distribución de becarios de programas de maestria por genero
becario_maestria.SEXO.value_counts(normalize=True).round(2)*100

# Se calcula la edad promedio para la categoria género
promedio_edad = becario_maestria.groupby("SEXO")["EDADBASES"].mean()
print(promedio_edad)


# Se calcula la distribución de becarios de programas de maestria por año
becario_maestria_año = pd.pivot_table(becario_maestria, values="ID_POSTULACION", index="AÑO_CONVOCATORIA", aggfunc="count")
becario_maestria_año.reset_index(inplace=True)

# Se transforman campos del dataframe pronabec_becario_año
becario_maestria_año["AÑO_CONVOCATORIA"] = pd.to_numeric(becario_maestria_año["AÑO_CONVOCATORIA"], errors="coerce")
becario_maestria_año["ID_POSTULACION"] = pd.to_numeric(becario_maestria_año["ID_POSTULACION"], errors="coerce")

# Se renombran atributos del dataframe pronabec_becario_año
becario_maestria_año.rename(columns=({"ID_POSTULACION":"CANTIDAD"}), inplace=True)

# Se organiza en función de año
becario_maestria_año = becario_maestria_año.sort_values("AÑO_CONVOCATORIA")


fig, ax = plt.subplots(figsize=(10, 6))

# Línea
ax.plot(
    becario_maestria_año["AÑO_CONVOCATORIA"],
    becario_maestria_año["CANTIDAD"],
    color="red",
    marker="o",
    linewidth=2.5,
    markersize=8
)

# Etiquetas
for x, y in zip(becario_maestria_año["AÑO_CONVOCATORIA"], becario_maestria_año["CANTIDAD"]):
    ax.annotate(
        f"{y:,}",
        (x, y),
        textcoords="offset points",
        xytext=(0, 8),
        ha="center",
        fontsize=10,
        #fontweight="bold"
    )

# Formato
#ax.set_title(
    #"Número de becarios por año de convocatoria",
    #fontsize=14,
    #fontweight="bold"
#)

ax.set_xlabel("Año")
ax.set_ylabel("Número de becarios de programas de maestría")

ax.set_xticks(becario_maestria_año["AÑO_CONVOCATORIA"])

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)

plt.tight_layout()
plt.show()


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
fig, ax = plt.subplots(figsize=(12, 6))

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
        xytext=(-12, -8),
        textcoords="offset points",
        fontsize=10,
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
        fontsize=10,
        color="#0B4F6C",
        fontweight="bold"
    )

    # Edad máxima
    ax.annotate(
        f'{row["EDAD_MAXIMA"]:.0f}',
        (row["AÑO_CONVOCATORIA"], row["EDAD_MAXIMA"]),
        xytext=(10, 6),
        textcoords="offset points",
        fontsize=10,
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
    bbox_to_anchor=(0.5, 1.08)
)

plt.tight_layout()
plt.show()

# Se calcula el sexo de los becarios de maestría
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
        fontsize=11
    )

    ax.text(
        año,
        fem + masc/2,
        f"{masc:.1f}%",
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
        fontsize=11
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



###############################################################################
# Se analiza los becarios de programas de doctorado
###############################################################################
becario_doctorado = pronabec_becario[pronabec_becario["NIVEL_EDUCATIVO"]=="DOCTORADO"]
becario_doctorado.shape

# Se analiza la distribución de becarios de programas de doctorado por genero
becario_doctorado.SEXO.value_counts(normalize=True).round(2)*100

# Se calcula la edad promedio para la categoria género
promedio_edad = becario_doctorado.groupby("SEXO")["EDADBASES"].mean()
print(promedio_edad)


# Se calcula la distribución de becarios de programas de doctorado por año
becario_doctorado_año = pd.pivot_table(becario_doctorado, values="ID_POSTULACION", index="AÑO_CONVOCATORIA", aggfunc="count")
becario_doctorado_año.reset_index(inplace=True)

# Se transforman campos del dataframe becario_becario_año
becario_doctorado_año["AÑO_CONVOCATORIA"] = pd.to_numeric(becario_doctorado_año["AÑO_CONVOCATORIA"], errors="coerce")
becario_doctorado_año["ID_POSTULACION"] = pd.to_numeric(becario_doctorado_año["ID_POSTULACION"], errors="coerce")

# Se renombran atributos del dataframe becario_becario_año
becario_doctorado_año.rename(columns=({"ID_POSTULACION":"CANTIDAD"}), inplace=True)

# Se organiza en función de año
becario_doctorado_año = becario_doctorado_año.sort_values("AÑO_CONVOCATORIA")


fig, ax = plt.subplots(figsize=(10, 6))

# Línea
ax.plot(
    becario_doctorado_año["AÑO_CONVOCATORIA"],
    becario_doctorado_año["CANTIDAD"],
    color="purple",
    marker="o",
    linewidth=2.5,
    markersize=8
)

# Etiquetas
for x, y in zip(becario_doctorado_año["AÑO_CONVOCATORIA"], becario_doctorado_año["CANTIDAD"]):
    ax.annotate(
        f"{y:,}",
        (x, y),
        textcoords="offset points",
        xytext=(0, 8),
        ha="center",
        fontsize=10,
        #fontweight="bold"
    )

# Formato
#ax.set_title(
    #"Número de becarios por año de convocatoria",
    #fontsize=14,
    #fontweight="bold"
#)

ax.set_xlabel("Año")
ax.set_ylabel("Número de becarios de programas de doctorado")

ax.set_xticks(becario_doctorado_año["AÑO_CONVOCATORIA"])

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)

plt.tight_layout()
plt.show()


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
fig, ax = plt.subplots(figsize=(12, 6))

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
        xytext=(-12, -8),
        textcoords="offset points",
        fontsize=10,
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
        fontsize=10,
        color="#0B4F6C",
        fontweight="bold"
    )

    # Edad máxima
    ax.annotate(
        f'{row["EDAD_MAXIMA"]:.0f}',
        (row["AÑO_CONVOCATORIA"], row["EDAD_MAXIMA"]),
        xytext=(10, 6),
        textcoords="offset points",
        fontsize=10,
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
    bbox_to_anchor=(0.5, 1.08)
)

plt.tight_layout()
plt.show()


# Se calcula el sexo de los becarios de doctorado
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
        fontsize=11
    )

    ax.text(
        año,
        fem + masc/2,
        f"{masc:.1f}%",
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
        fontsize=11
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


##############################################################################
# Se analiza el pais de destino de los becarios de programas de maestria y doctorado
# considerando los dataframes becario_maestria y becario_doctorado
###############################################################################

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

plt.figure(figsize=(12, 6))

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
        pct + 0.5,
        i,
        f"{pct:.1f}% ({total:,})",
        va="center",
        fontsize=14
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

plt.figure(figsize=(12, 6))

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
        pct + 0.5,
        i,
        f"{pct:.1f}% ({total:,})",
        va="center",
        fontsize=14
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


plt.figure(figsize=(12, 6))

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
        pct + 0.2,
        i,
        f"{pct:.1f}% ({total:,})",
        va="center",
        fontsize=14
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


plt.figure(figsize=(12, 6))

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
        fontsize=14
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

# Se convierte el archivo Base_carreras_STEM en un dataframe, y se le denomina
# stem

stem = pd.read_excel("Base_carreras_STEM.xlsx", sheet_name="Sheet1", header=0)
stem.columns
stem = stem[["ID", "STEM", "AREA_STEM_CIP"]]
stem.rename(columns=({"ID":"ID_POSTULACION"}), inplace=True)
stem.shape
stem["ID_POSTULACION"] = stem["ID_POSTULACION"].astype("string")


# Se usa también el dataframe pronabec_becario
pronabec_becario.columns
pronabec_becario.shape

fusion = pd.merge(pronabec_becario, stem, on="ID_POSTULACION", how="right")

# Considerando el dataframe fusion, se analiza la prevalencia de las carreras
# STEM para maestria

fusion_maestria = fusion[fusion["NIVEL_EDUCATIVO"]=="MAESTRIA"]
maes_stem = fusion_maestria.STEM.value_counts(normalize=True).round(2)*100
maes_stem = maes_stem.to_frame()
maes_stem.reset_index(inplace=True)
maes_stem.rename(columns=({"proportion":"porcentaje"}), inplace=True)

labels = maes_stem["STEM"]
sizes = maes_stem["porcentaje"]

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

barra_stem_maestria = fusion_maestria[fusion_maestria["STEM"]=="STEM"]
barra_maestria = barra_stem_maestria.AREA_STEM_CIP.value_counts(normalize=True).round(2)*100
barra_maestria = barra_maestria.to_frame()
barra_maestria.reset_index(inplace=True)

area_map = {
    "E": "Engineering",
    "S": "Science",
    "T": "Technology",
    "M": "Mathematics"
}

barra_maestria["AREA_STEM_CIP"] = barra_maestria["AREA_STEM_CIP"].replace(area_map)


fig, ax = plt.subplots(figsize=(10, 6))

barras = ax.bar(
    barra_maestria["AREA_STEM_CIP"],
    barra_maestria["proportion"],
    color="#0B4F6C",
    edgecolor="white",
    linewidth=1.5
)

# Etiquetas arriba de cada columna
for barra in barras:
    alto = barra.get_height()
    ax.text(
        barra.get_x() + barra.get_width() / 2,
        alto,
        f"{alto:,.0f}",
        ha="center",
        va="bottom",
        fontsize=12,
        #fontweight="bold",
        color="black"
    )


#ax.set_title(
    #"Evolución del número total de postulantes, 2013-2025",
    #fontsize=14,
    #fontweight="bold"
#)

plt.xlabel("Categoria STEM")
plt.ylabel("Porcentaje (%)")


ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()


# Considerando el dataframe fusion, se analiza la prevalencia de las carreras
# STEM para doctorado

fusion_doctorado = fusion[fusion["NIVEL_EDUCATIVO"]=="DOCTORADO"]
doct_stem = fusion_doctorado.STEM.value_counts(normalize=True).round(2)*100
doct_stem = doct_stem.to_frame()
doct_stem.reset_index(inplace=True)
doct_stem.rename(columns=({"proportion":"porcentaje"}), inplace=True)

labels = doct_stem["STEM"]
sizes = doct_stem["porcentaje"]

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


barra_stem_doctorado = fusion_doctorado[fusion_doctorado["STEM"]=="STEM"]
barra_doctorado = barra_stem_doctorado.AREA_STEM_CIP.value_counts(normalize=True).round(2)*100
barra_doctorado = barra_doctorado.to_frame()
barra_doctorado.reset_index(inplace=True)

area_map = {
    "E": "Engineering",
    "S": "Science",
    "T": "Technology",
    "M": "Mathematics"
}

barra_doctorado["AREA_STEM_CIP"] = barra_doctorado["AREA_STEM_CIP"].replace(area_map)


fig, ax = plt.subplots(figsize=(10, 6))

barras = ax.bar(
    barra_doctorado["AREA_STEM_CIP"],
    barra_doctorado["proportion"],
    color="#0B4F6C",
    edgecolor="white",
    linewidth=1.5
)

# Etiquetas arriba de cada columna
for barra in barras:
    alto = barra.get_height()
    ax.text(
        barra.get_x() + barra.get_width() / 2,
        alto,
        f"{alto:,.0f}",
        ha="center",
        va="bottom",
        fontsize=12,
        #fontweight="bold",
        color="black"
    )


#ax.set_title(
    #"Evolución del número total de postulantes, 2013-2025",
    #fontsize=14,
    #fontweight="bold"
#)

plt.xlabel("Categoria STEM")
plt.ylabel("Porcentaje (%)")


ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()

###############################################################################
# Se analiza el cumplimiento del compomiso de servicio (CSP), de forma global, por
# maestria y doctorado
###############################################################################

# Tabla resumen
csp = (
    pronabec_becario["ESTADO_DE_CSP"]
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
    "SIN INFORMACIÓN",
    "INCUMPLIMIENTO"
]

# Colores fijos por estado
colores_estado = {
    "CUMPLIO": "#0B4F6C",
    "PENDIENTE": "#5FB7C6",
    "SIN INFORMACIÓN": "#A3AD2C",
    "INCUMPLIMIENTO": "#D9D9D9"
}

# Etiquetas amigables
etiquetas_estado = {
    "CUMPLIO": "Cumplió",
    "PENDIENTE": "Pendiente",
    "SIN INFORMACIÓN": "Sin información",
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
# Se analiza el estado de los becarios de maestria y doctorado del 2022 al 2025
###############################################################################

segunda = pronabec_becario[(pronabec_becario["AÑO_CONVOCATORIA"] >= 2022) & (pronabec_becario["AÑO_CONVOCATORIA"] <= 2025)]
segunda.columns

segunda.NIVEL_EDUCATIVO.value_counts()


# Se renombran algunos registros de la columna INSTITUCION_ORIGEN_PREGRADO del dataframe segunda
segunda["INSTITUCION_ORIGEN_PREGRADO"] = segunda["INSTITUCION_ORIGEN_PREGRADO"].replace({
    "PONTIFICIA UNIVERSIDAD CATOLICA DEL PERU":
    "PONTIFICIA UNIVERSIDAD CATÓLICA DEL PERÚ"
})


# Se renombran algunos registros de la columna REGION_PREGRADO del dataframe segunda
segunda.REGION_PREGRADO.value_counts()
segunda["REGION_PREGRADO"] = segunda["REGION_PREGRADO"].replace({
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
   
    
# Se analiza la formación académica de los becarios de maestria
segunda_maestria = segunda[segunda["NIVEL_EDUCATIVO"]=="MAESTRIA"]
segunda_maestria_ima = segunda_maestria.GRADO_PREGRADO.value_counts(normalize=True).round(3)*100
segunda_maestria_ima = segunda_maestria_ima.to_frame()
segunda_maestria_ima.reset_index(inplace=True)
segunda_maestria_ima.rename(columns=({"proportion":"porcentaje"}), inplace=True)
segunda_maestria_ima["porcentaje"] = segunda_maestria_ima["porcentaje"].astype("float")

fig, ax = plt.subplots(figsize=(10, 6))

barras = ax.bar(
    segunda_maestria_ima["GRADO_PREGRADO"],
    segunda_maestria_ima["porcentaje"],
    color="#2F6690",
    edgecolor="white",
    linewidth=1.5
)

# Etiquetas arriba de cada columna
for barra in barras:
    alto = barra.get_height()
    ax.text(
        barra.get_x() + barra.get_width() / 2,
        alto,
        f"{alto:,.1f}",
        ha="center",
        va="bottom",
        fontsize=14,
        #fontweight="bold",
        color="black"
    )


#ax.set_title(
    #"Evolución del número total de postulantes, 2013-2025",
    #fontsize=14,
    #fontweight="bold"
#)

plt.xlabel("Grado académico")
plt.ylabel("Porcentaje (%)")


ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()


# Se analiza la formación académica de los becarios de doctorado

segunda_doctorado = segunda[segunda["NIVEL_EDUCATIVO"]=="DOCTORADO"]
segunda_doctorado_ima = (
    segunda_doctorado["GRADO_PREGRADO"]
    .value_counts(normalize=True)
    .mul(100)
    .to_frame(name="porcentaje")
    .reset_index()
)



fig, ax = plt.subplots(figsize=(10, 6))

barras = ax.bar(
    segunda_doctorado_ima["GRADO_PREGRADO"],
    segunda_doctorado_ima["porcentaje"],
    color="#2F6690",
    edgecolor="white",
    linewidth=1.5
)

# Etiquetas arriba de cada columna
for barra in barras:
    alto = barra.get_height()
    ax.text(
        barra.get_x() + barra.get_width() / 2,
        alto,
        f"{alto:,.3f}",
        ha="center",
        va="bottom",
        fontsize=14,
        #fontweight="bold",
        color="black"
    )


#ax.set_title(
    #"Evolución del número total de postulantes, 2013-2025",
    #fontsize=14,
    #fontweight="bold"
#)

plt.xlabel("Grado académico")
plt.ylabel("Porcentaje (%)")


ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()

###############################################################################
# Se analiza el tipo de gestión de la universdad del pregrado de becarios tanto
# de programas de maestria como doctorado
###############################################################################

# Becarios de programas de maestria

gestion_maestria = segunda_maestria.GESTION_PREGRADO.value_counts(normalize=True).round(2)*100
gestion_maestria = gestion_maestria.to_frame()
gestion_maestria.reset_index(inplace=True)
gestion_maestria.rename(columns=({"proportion":"porcentaje"}), inplace=True)
gestion_maestria.columns


labels = gestion_maestria["GESTION_PREGRADO"]
sizes = gestion_maestria["porcentaje"]

# Colores institucionales
colors = ["#264653", "#8FAF3C"]

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
        "fontsize": 12,
        "fontweight": "bold"
    }
)

# Estilo de porcentajes
for autotext in autotexts:
    autotext.set_color("white")
    autotext.set_fontsize(13)
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


# Becarios de programas de doctorado

gestion_doctorado = segunda_doctorado.GESTION_PREGRADO.value_counts(normalize=True).round(2)*100
gestion_doctorado = gestion_doctorado.to_frame()
gestion_doctorado.reset_index(inplace=True)
gestion_doctorado.rename(columns=({"proportion":"porcentaje"}), inplace=True)
gestion_doctorado.columns


labels = gestion_doctorado["GESTION_PREGRADO"]
sizes = gestion_doctorado["porcentaje"]

# Colores institucionales
colors = ["#264653", "#8FAF3C"]

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
        "fontsize": 12,
        "fontweight": "bold"
    }
)

# Estilo de porcentajes
for autotext in autotexts:
    autotext.set_color("white")
    autotext.set_fontsize(13)
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
# Se analiza la universidad de procedencia (Top 10) de los becarios de programa
# de maestria y doctorado
###############################################################################

# Se realiza el análisis para becarios de programas de maestría
total = segunda_maestria["INSTITUCION_ORIGEN_PREGRADO"].value_counts()

proporcion = (
    segunda_maestria["INSTITUCION_ORIGEN_PREGRADO"]
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
        fontsize=14
    )

plt.xlabel("Participación (%)")

plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

plt.grid(axis="x", linestyle="--", alpha=0.3)

plt.tight_layout()
plt.show()


# Se realiza el análisis para becarios de programas de doctorado
total = segunda_doctorado["INSTITUCION_ORIGEN_PREGRADO"].value_counts()

proporcion = (
    segunda_doctorado["INSTITUCION_ORIGEN_PREGRADO"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

pais_segunda_doctorado = pd.DataFrame({
    "TOTAL": total,
    "PROPORCION (%)": proporcion
})

pais_segunda_doctorado
pais_segunda_doctorado.reset_index(inplace=True)
pais_segunda_doctorado = pais_segunda_doctorado.head(10)
pais_segunda_doctorado.columns

pais_segunda_doctorado = pais_segunda_doctorado.sort_values("TOTAL", ascending=True)

color_principal = "#0B4F6C"   # azul institucional
color_secundario = "#5FB7C6"  # celeste

plt.figure(figsize=(12, 6))

bars = plt.barh(
    pais_segunda_doctorado["INSTITUCION_ORIGEN_PREGRADO"],
    pais_segunda_doctorado["PROPORCION (%)"],
    color=color_principal
)

# Etiquetas: porcentaje + total
for i, (pct, total) in enumerate(
    zip(
        pais_segunda_doctorado["PROPORCION (%)"],
        pais_segunda_doctorado["TOTAL"]
    )
):
    plt.text(
        pct + 0.5,
        i,
        f"{pct:.1f}% ({total:,})",
        va="center",
        fontsize=14
    )

plt.xlabel("Participación (%)")

plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

plt.grid(axis="x", linestyle="--", alpha=0.3)

plt.tight_layout()
plt.show()


###############################################################################
# Se analiza las regiones de los becarios de programas de maestria y doctorado
###############################################################################

# Becarios de programas de maestria

peru_maestria = segunda_maestria.REGION_PREGRADO.value_counts()
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

# no considera el sin registro
caso = peru_maestria[peru_maestria["Region"]!="SIN REGISTRO"]
caso["cantidad"].sum()



# Becarios de programas de doctorado

peru_doctorado = segunda_doctorado.REGION_PREGRADO.value_counts()
peru_doctorado = peru_doctorado.to_frame()
peru_doctorado.reset_index(inplace=True)
peru_doctorado.rename(columns=({"REGION_PREGRADO":"Region", "count":"cantidad"}), inplace=True)


#código para la elaboración de un mapa de calor, considerando una variable específica, utilizando geopandas
# Se construye el mapa de Perú, a nivel departamental
peru = geopandas.read_file("INEI_LIMITE_DEPARTAMENTAL_GEOGPSPERU_JUANSUYO_931381206.shp")
peru.columns
peru.rename(columns=({"NOMBDEP":"Region"}), inplace=True)

# Se fusiona el dataframe peru con region
peru = pd.merge(peru,peru_doctorado, on="Region", how="left")


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
        color_texto = "white" if row["cantidad"] >= 14 else "black"

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

# no considera el sin registro
caso = peru_doctorado[peru_doctorado["Region"]!="SIN REGISTRO"]
caso["cantidad"].sum()


###############################################################################
# Modelo predictivo para predicir el cumplimiento de becarios
###############################################################################

pronabec_becario.columns
modelo = pronabec_becario[pronabec_becario["ESTADO_DE_CSP"]=="CUMPLIO"]

cumplidor = modelo[["ESTADO_DE_CSP","SEXO", "EDADBASES", "NIVEL_EDUCATIVO", "PAISDESTINO", "INSTITUCION"]]

# =========================
# 1. Resumen general
# =========================
total_cumplidores = len(cumplidor)

print(f"Total de cumplidores: {total_cumplidores:,}")

# =========================
# 2. Perfil de edad
# =========================
perfil_edad = cumplidor["EDADBASES"].describe()

print(perfil_edad)

# =========================
# 3. Función para tablas de perfil
# =========================
def tabla_perfil(data, columna):
    tabla = (
        data[columna]
        .value_counts()
        .reset_index()
    )
    
    tabla.columns = [columna, "TOTAL"]
    
    tabla["PROPORCION (%)"] = (
        tabla["TOTAL"] / tabla["TOTAL"].sum() * 100
    ).round(1)
    
    return tabla

# =========================
# 4. Perfil categórico
# =========================
perfil_sexo = tabla_perfil(cumplidor, "SEXO")
perfil_nivel = tabla_perfil(cumplidor, "NIVEL_EDUCATIVO")
perfil_pais = tabla_perfil(cumplidor, "PAISDESTINO")
perfil_institucion = tabla_perfil(cumplidor, "INSTITUCION")

# Mostrar resultados
print("\nPerfil por sexo")
print(perfil_sexo)

print("\nPerfil por nivel educativo")
print(perfil_nivel)

print("\nPerfil por país de destino")
print(perfil_pais.head(10))

print("\nPerfil por institución")
print(perfil_institucion.head(10))

perfil_cumplidor = {
    "Total cumplidores": len(cumplidor),
    "Edad promedio": round(cumplidor["EDADBASES"].mean(), 1),
    "Edad mediana": round(cumplidor["EDADBASES"].median(), 1),
    "Sexo predominante": cumplidor["SEXO"].mode()[0],
    "Nivel predominante": cumplidor["NIVEL_EDUCATIVO"].mode()[0],
    "País destino principal": cumplidor["PAISDESTINO"].mode()[0],
    "Institución principal": cumplidor["INSTITUCION"].mode()[0]
}

perfil_cumplidor = pd.DataFrame(
    perfil_cumplidor.items(),
    columns=["Indicador", "Valor"]
)

perfil_cumplidor


###############################################################################
# Modelo de regresión logística
###############################################################################

# Se construye un modelo predictivo para predecir la probabilidad de cumplimiento de becarios
# utilizando el dataframe fusion, que representa el merge entre stem y pronabec_becario
fusion.shape
fusion.columns
fusion.info()

# Se cuenta la cantidad de registros que tiene el campo STEM del dataframe fusion
fusion["STEM"].count()

# Se analiza la distribución de la condición STEM de los becarios tanto de programas de maestría como doctorado
fusion.STEM.value_counts()
fusion.STEM.value_counts(normalize=True).round(2)*100

# Se analiza la distribución del campo ESTADO_DE_CSP
fusion.ESTADO_DE_CSP.value_counts(normalize=True).round(3)*100

# Se crea explicitamente la categoría "NO CUMPLIÓ", y se construye una variable booleana
fusion["CUMPLIMIENTO_CAT"] = fusion["ESTADO_DE_CSP"].replace(
    {
        "PENDIENTE": "NO CUMPLIO",
        "SIN INFORMACIÓN": "NO CUMPLIO",
        "INCUMPLIMIENTO": "NO CUMPLIO"
    }
)


fusion["CUMPLIO_BIN"] = fusion["CUMPLIMIENTO_CAT"].map(
    {
        "CUMPLIO": 1,
        "NO CUMPLIO": 0
    }
)


fusion.columns

# En el caso de Pais de destino se agrupan por cantidad de becarios
top_paises = (
    fusion["PAISDESTINO"]
    .value_counts()
    .head(10)
    .index
)

fusion["PAISDESTINO_GRP"] = np.where(
    fusion["PAISDESTINO"].isin(top_paises),
    fusion["PAISDESTINO"],
    "OTROS"
)


# Se construye el dataset que permitirá hacer el análisis
modelo = fusion[["CUMPLIO_BIN", "EDADBASES", "SEXO", "STEM", "NIVEL_EDUCATIVO", "PAISDESTINO_GRP"]]


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

# Dado que mi objetivo es estimar la probabilidad de cumplimiento de becarios en el extranjero
# entonces tengo que deliminar la poblicación de mi estudio
df_model = df_model[
    df_model["PAISDESTINO_GRP"] != "PERU"
].copy()



df_model["CUMPLIO_BIN"] = df_model["CUMPLIO_BIN"].astype(int)
df_model["EDADBASES"] = pd.to_numeric(df_model["EDADBASES"], errors="coerce")
df_model = df_model.dropna()


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















