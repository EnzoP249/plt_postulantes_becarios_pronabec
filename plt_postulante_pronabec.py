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

pronabec = pd.read_excel("0_BGB_2013_2025.xlsx", sheet_name="Sheet1", header=0, converters=converters)

# Se identifican caracteristicas estructurales del dataframe pronabec
pronabec.shape
pronabec.info()
pronabec.dtypes
pronabec.columns



# Se reorganiza la información que integra el objeto dataframe pronabec
pronabec = pronabec[["N","ID_POSTULACION", "CONDICION_FINAL", "NEXPEDIENTE", "AÑO_CONVOCATORIA",
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


###############################################################################
# Se analiza los becarios de programas de doctorado
###############################################################################
becario_doctorado = pronabec_becario[pronabec_becario["NIVEL_EDUCATIVO"]=="DOCTORADO"]

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

##############################################################################
# Se analiza el pais de destino de los becarios de programas de maestria y doctorado
# considerando los dataframes becario_maestria y becario_doctorado
###############################################################################

# Se realiza el análisis para becarios de programas de maestría

pais_maestria = becario_maestria.PAISDESTINO.value_counts(normalize=True).round(2)*100
pais_maestria = pais_maestria.to_frame()
pais_maestria.reset_index(inplace=True)
pais_maestria.rename(columns=({"proportion":"Porcentaje (%)"}), inplace=True)
pais_maestria = pais_maestria.head(10)
pais_maestria.columns

pais_maestria = pais_maestria.sort_values("Porcentaje (%)", ascending=True)

color_principal = "#0B4F6C"   # azul institucional
color_secundario = "#5FB7C6"  # celeste

colors = [color_secundario] * len(pais_maestria)
colors[-1] = color_principal  # destacar el mayor

plt.figure(figsize=(12, 6))

bars = plt.barh(pais_maestria["PAISDESTINO"], pais_maestria["Porcentaje (%)"], color=colors)

# Etiquetas
for i, v in enumerate(pais_maestria["Porcentaje (%)"]):
    plt.text(v + 0.5, i, f"{v:.0f}%", va="center", fontsize=10)

# Estética consultoría
plt.xlabel("Porcentaje (%)")
#plt.title("Distribución de subvenciones por tipo de intervención", fontsize=13)

plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

plt.grid(axis="x", linestyle="--", alpha=0.3)

plt.tight_layout()
plt.show()


# Se realiza el análisis para becarios de programas de doctorado

pais_doctorado = becario_doctorado.PAISDESTINO.value_counts(normalize=True).round(2)*100
pais_doctorado = pais_doctorado.to_frame()
pais_doctorado.reset_index(inplace=True)
pais_doctorado.rename(columns=({"proportion":"Porcentaje (%)"}), inplace=True)
pais_doctorado = pais_doctorado.head(10)
pais_doctorado.columns

pais_doctorado = pais_doctorado.sort_values("Porcentaje (%)", ascending=True)

color_principal = "#0B4F6C"   # azul institucional
color_secundario = "#5FB7C6"  # celeste

colors = [color_secundario] * len(pais_doctorado)
colors[-1] = color_principal  # destacar el mayor

plt.figure(figsize=(12, 6))

bars = plt.barh(pais_doctorado["PAISDESTINO"], pais_doctorado["Porcentaje (%)"], color=colors)

# Etiquetas
for i, v in enumerate(pais_doctorado["Porcentaje (%)"]):
    plt.text(v + 0.5, i, f"{v:.0f}%", va="center", fontsize=10)

# Estética consultoría
plt.xlabel("Porcentaje (%)")
#plt.title("Distribución de subvenciones por tipo de intervención", fontsize=13)

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

institucion_maestria = becario_maestria.INSTITUCION.value_counts(normalize=True).round(2)*100
institucion_maestria = institucion_maestria.to_frame()
institucion_maestria.reset_index(inplace=True)
institucion_maestria.rename(columns=({"proportion":"Porcentaje (%)"}), inplace=True)
institucion_maestria = institucion_maestria.head(10)
institucion_maestria.columns

institucion_maestria = institucion_maestria.sort_values("Porcentaje (%)", ascending=True)

color_principal = "#0B4F6C"   # azul institucional
color_secundario = "#5FB7C6"  # celeste

colors = [color_secundario] * len(institucion_maestria)
colors[-1] = color_principal  # destacar el mayor

plt.figure(figsize=(12, 6))

bars = plt.barh(institucion_maestria["INSTITUCION"], institucion_maestria["Porcentaje (%)"], color=colors)

# Etiquetas
for i, v in enumerate(institucion_maestria["Porcentaje (%)"]):
    plt.text(v + 0.5, i, f"{v:.0f}%", va="center", fontsize=10)

# Estética consultoría
plt.xlabel("Porcentaje (%)")
#plt.title("Distribución de subvenciones por tipo de intervención", fontsize=13)

plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

plt.grid(axis="x", linestyle="--", alpha=0.3)

plt.tight_layout()
plt.show()


# Se realiza el análisis para becarios de programas de doctorado

institucion_doctorado = becario_doctorado.INSTITUCION.value_counts(normalize=True).round(2)*100
institucion_doctorado = institucion_doctorado.to_frame()
institucion_doctorado.reset_index(inplace=True)
institucion_doctorado.rename(columns=({"proportion":"Porcentaje (%)"}), inplace=True)
institucion_doctorado = institucion_doctorado.head(10)
institucion_doctorado.columns

institucion_doctorado = institucion_doctorado.sort_values("Porcentaje (%)", ascending=True)

color_principal = "#0B4F6C"   # azul institucional
color_secundario = "#5FB7C6"  # celeste

colors = [color_secundario] * len(institucion_maestria)
colors[-1] = color_principal  # destacar el mayor

plt.figure(figsize=(12, 6))

bars = plt.barh(institucion_doctorado["INSTITUCION"], institucion_doctorado["Porcentaje (%)"], color=colors)

# Etiquetas
for i, v in enumerate(institucion_doctorado["Porcentaje (%)"]):
    plt.text(v + 0.5, i, f"{v:.0f}%", va="center", fontsize=10)

# Estética consultoría
plt.xlabel("Porcentaje (%)")
#plt.title("Distribución de subvenciones por tipo de intervención", fontsize=13)

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
# Se analiza el cumplimiento del compomiso de servicio, de forma global, por
# maestria y doctorado
###############################################################################

csp = pronabec_becario.ESTADO_DE_CSP.value_counts(normalize=True).round(2)*100
csp = csp.to_frame()
csp.reset_index(inplace=True)
csp.rename(columns=({"proportion":"porcentaje"}), inplace=True)
csp.columns

labels = csp["ESTADO_DE_CSP"]
sizes = csp["porcentaje"]

# Colores institucionales
colors = [
    "#0B4F6C",  # Azul petróleo
    "#5FB7C6",  # Celeste institucional
    "#A3AD2C",  # Verde oliva
    "#D9D9D9"   # Gris neutro
]

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


csp = becario_maestria.ESTADO_DE_CSP.value_counts(normalize=True).round(2)*100
csp = csp.to_frame()
csp.reset_index(inplace=True)
csp.rename(columns=({"proportion":"porcentaje"}), inplace=True)
csp.columns

labels = csp["ESTADO_DE_CSP"]
sizes = csp["porcentaje"]

# Colores institucionales
colors = [
    "#0B4F6C",  # Azul petróleo
    "#5FB7C6",  # Celeste institucional
    "#A3AD2C",  # Verde oliva
    "#D9D9D9"   # Gris neutro
]

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


csp = becario_doctorado.ESTADO_DE_CSP.value_counts(normalize=True).round(2)*100
csp = csp.to_frame()
csp.reset_index(inplace=True)
csp.rename(columns=({"proportion":"porcentaje"}), inplace=True)
csp.columns

labels = csp["ESTADO_DE_CSP"]
sizes = csp["porcentaje"]

# Colores institucionales
colors = [
    "#0B4F6C",  # Azul petróleo
    "#5FB7C6",  # Celeste institucional
    "#A3AD2C",  # Verde oliva
    "#D9D9D9"   # Gris neutro
]

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







