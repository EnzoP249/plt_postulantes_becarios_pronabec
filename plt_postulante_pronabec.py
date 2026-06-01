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

# En total, durante el periodo de análisis, ¿cuántos postulantes recibieron la beca?
pronabec_becario = pronabec[pronabec["CONDICION_FINAL"]=="SE LE ADJUDICÓ LA BECA"]
a = pronabec_becario["ID_POSTULACION"].count()
print(f"Los postulantes que recibieron la beca, durante el periodo de análisis, fueron {a}, lo que representa el 19%")

# Se cuentan los identificadores de postulaciones únicas considerando el dataframe pronabec_becario
pronabec_becario["ID_POSTULACION"].nunique()

###############################################################################
# Caracteristicas generales
###############################################################################
pronabec_becario["ID_POSTULACION"].count()

# Considerando el dataframe pronabec_becario, se obtiene una distribución por nivel educativo, a nivel general
pronabec_becario.NIVEL_EDUCATIVO.value_counts(normalize=True).round(2)*100

# Se analiza el cumplimiento del compomiso de servicio
pronabec_becario.ESTADO_DE_CSP.value_counts(normalize=True).round(2)*100


###############################################################################
# Considerando el dataframe pronabec_becario, se obtiene la evolución de los becarios, a nivel temporal
###############################################################################

pronabec_becario_año = pd.pivot_table(pronabec_becario, values="ID_POSTULACION", index="AÑO_CONVOCATORIA", aggfunc="count")
pronabec_becario_año.reset_index(inplace=True)

# Se transforman campos del dataframe pronabec_becario_año
pronabec_becario_año["AÑO_CONVOCATORIA"] = pd.to_numeric(pronabec_becario_año["AÑO_CONVOCATORIA"], errors="coerce")
pronabec_becario_año["ID_POSTULACION"] = pd.to_numeric(pronabec_becario_año["ID_POSTULACION"], errors="coerce")

# Se renombran atributos del dataframe pronabec_becario_año
pronabec_becario_año.rename(columns=({"ID_POSTULACION":"CANTIDAD"}), inplace=True)

# Se organiza en función de año
pronabec_becario_año = pronabec_becario_año.sort_values("CANTIDAD")

fig, ax = plt.subplots(figsize=(10, 6))

barras = ax.bar(
    pronabec_becario_año["AÑO_CONVOCATORIA"],
    pronabec_becario_año["CANTIDAD"],
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
ax.set_xticks(pronabec_becario_año["AÑO_CONVOCATORIA"])
ax.set_xticklabels(
    pronabec_becario_año["AÑO_CONVOCATORIA"],
    rotation=0
)

ax.set_title(
    "Evolución del número total de becarios, 2013-2025",
    fontsize=14,
    fontweight="bold"
)
plt.xlabel("Año")
plt.ylabel("Número de becarios")


ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()

###############################################################################
# Considerando el dataframe pronabec_becario, se obtiene la evolución de los
# becarios por nivel educativo a nivel temporal
###############################################################################

pronabec_becario_edu = pd.pivot_table(pronabec_becario, values="ID_POSTULACION", index="AÑO_CONVOCATORIA", columns="NIVEL_EDUCATIVO", aggfunc="count")
pronabec_becario_edu.reset_index(inplace=True)

# ==========================================
# PREPARACIÓN DE DATOS
# ==========================================
df_plot = pronabec_becario_edu.fillna(0)

# ==========================================
# GRÁFICO
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6))

# Barra inferior: Doctorado
barras_doc = ax.bar(
    df_plot["AÑO_CONVOCATORIA"],
    df_plot["DOCTORADO"],
    color="#4C72B0",
    label="Doctorado"
)

# Barra superior: Maestría
barras_mae = ax.bar(
    df_plot["AÑO_CONVOCATORIA"],
    df_plot["MAESTRIA"],
    bottom=df_plot["DOCTORADO"],
    color="#DD8452",
    label="Maestría"
)

# ==========================================
# ETIQUETAS INTERNAS
# ==========================================
for _, row in df_plot.iterrows():

    año = row["AÑO_CONVOCATORIA"]
    doctorado = row["DOCTORADO"]
    maestria = row["MAESTRIA"]

    # Doctorado
    if doctorado > 0:
        ax.text(
            año,
            doctorado / 2,
            f"{int(doctorado)}",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white"
        )

    # Maestría
    if maestria > 0:
        ax.text(
            año,
            doctorado + maestria / 2,
            f"{int(maestria)}",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white"
        )

# ==========================================
# TOTAL SOBRE CADA BARRA
# ==========================================
for _, row in df_plot.iterrows():

    total = row["DOCTORADO"] + row["MAESTRIA"]

    ax.text(
        row["AÑO_CONVOCATORIA"],
        total + total * 0.02,
        f"{int(total):,}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color="black"
    )

# ==========================================
# FORMATO
# ==========================================
ax.set_title(
    "Número de becarios por nivel de estudio, 2013 - 2025",
    fontsize=14,
    fontweight="bold"
)

ax.set_xlabel("Año")
ax.set_ylabel("Cantidad de becarios")

ax.set_xticks(df_plot["AÑO_CONVOCATORIA"])

ax.legend(frameon=False)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Espacio adicional para los totales
ax.set_ylim(
    0,
    (df_plot["DOCTORADO"] + df_plot["MAESTRIA"]).max() * 1.15
)

plt.tight_layout()
plt.show()















