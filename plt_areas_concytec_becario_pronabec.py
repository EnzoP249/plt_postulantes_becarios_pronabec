# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 11:16:56 2026

@author: Enzo
"""

###############################################################################
# PROYECTO PARA UN ESQUEMA DE MATCHING USANDO BECARIOS DEL PRONABEC
###############################################################################

###############################################################################
# OBJETIVO: ESTE PROYECTO TIENE COMO FINALIDADES IMPLEMENTAR UN ALGORITMO SEMANTICO
# ENTRE PROGRAMA DE MAESTRIA Y DOCTORADO CON AREAS PRIORITARIAS DEFINIDAS POR EL
# CONCYTEC; Y ELABORAR UN ANALISIS DE LA INFORMACIÓN OBTENIDA
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


# Se renombra el atributo NOBRECARRERA por CARRERA_PRONABEC
pronabec.rename(columns=({"NOMBRECARRERA":"CARRERA_PRONABEC"}), inplace=True)

###############################################################################
# Se diseña e implementa un algoritmo de Clasificación Determinista 
# basado en Léxico y Reglas jerarquicas
###############################################################################

# 1. Normalización estricta de caracteres y símbolos
def normalizar_texto(texto):
    if not isinstance(texto, str):
        return ""
    # Quitar tildes y acentos diacríticos
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join([c for c in texto if unicodedata.category(c) != "Mn"])
    # Reemplazar símbolos y puntuación por espacios para no romper límites de palabra
    texto = re.sub(r"[^A-Z0-9\s]", " ", texto.upper())
    return " ".join(texto.split())


# 2. Diccionario Síntesis: Frases compuestas específicas + Raíces Multilingües
REGLAS_MATCHING_SINTESIS = {
    "Biotecnología": [
        # Frases compuestas específicas (ES)
        r"\bBIOTECNOLOGIA BIODIVERSA\b",
        r"\bBIOTECNOLOGIA INDUSTRIAL\b",
        r"\bBIOENERGIA\b",
        r"\bBIORREMEDIACION\b",
        r"\bBIOTECNOLOGIA SANITARIA\b",
        r"\bBIOTECNOLOGIA AGROPECUARIA\b",
        r"\bBIOTECNOLOGIA PESQUERA\b",
        r"\bBIOTECNOLOGIA ACUICOLA\b",
        r"\bTECNOLOGIA AGROALIMENTARIA\b",
        # Raíces comodín multilingües (Capturan plurales, derivados e idiomas EN/FR/PT/DE)
        r"\bBIOTECNOL",  # ES, PT: Biotecnología, Biotecnológica
        r"\bBIOTECHNOL",  # EN, FR, DE: Biotechnology, Biotechnologie
        r"\bBIOTECH",  # Abreviatura internacional
        r"\bBIOENGINEER",  # EN: Bioengineering
        r"\bBIOREMED",  # EN, FR: Bioremediation
        r"\bAGRI FOOD TECH",  # EN
        r"\bBIOQUIMIC",  # ES, PT: Bioquímica
        r"\bBIOCHEMIS",  # EN, DE: Biochemistry
        r"\bBIOCHIMIE",  # FR
    ],
    "Nanotecnología": [
        # Frases compuestas
        r"\bMATERIALES AVANZADOS\b",
        # Raíces comodín multilingües (Capturan Nanotecnología, Nanotech, Nanociencias, Nanomateriales, Nanoestructuras)
        r"\bNANOTEC",  # ES, PT
        r"\bNANOTECHNOL",  # EN, FR, DE
        r"\bNANOTECH",  # Internacional
        r"\bNANOCIEN",  # ES, PT
        r"\bNANOSCIENC",  # EN, FR
        r"\bNANOSCIENZ",  # IT
        r"\bNANOWISSENSCHAFT",  # DE
        r"\bNANOMAT",  # ES, PT, EN, FR
        r"\bNANOESTR",  # ES, PT
        r"\bNANOSTRUCT",  # EN, FR
    ],
    "Ambiente y Cambio Climático": [
        # 1. Ecoeficiencia y Reducción de Emisiones (Texto específico)
        r"\bTECNOLOGIAS? ECOEFICIENTE",
        r"\bAPROVECHAMIENTO SOSTENIBLE DE RECURSOS\b",
        r"\bREDUCCION DE EMISIONES\b",
        r"\bECOEFICIEN",
        r"\bECOEFFICIEN",
        # 2. Sistemas Energéticos e Integración Inteligente (Texto específico)
        r"\bSISTEMAS? ENERGETICOS? INTELIGENTE",
        r"\bDIGITALIZACION ENERGETICA\b",
        r"\bTRANSICION ENERGETICA\b",
        r"\bEFICIENCIA ENERGETICA\b",
        r"\bDESCARBONIZACION\b",
        r"\bENERGIA RENOVABLE",
        r"\bRENEWABLE ENERGY",
        r"\bENERGIA LIMPIA",
        r"\bCLEAN ENERGY",
        # 3. Cambio Climático y Resiliencia
        r"\bCAMBIO CLIMATICO\b",
        r"\bCLIMATE CHANGE\b",
        r"\bCHANGEMENT CLIMATIQUE\b",
        r"\bMUDANCA CLIMATICA\b",
        r"\bKLIMAWANDEL\b",
        r"\bRESILIENCIA CLIMATICA\b",
        # 4. Desarrollo Sostenible y Ciencias Aplicadas
        r"\bDESARROLLO SOSTENIBLE\b",
        r"\bSUSTAINABLE DEVELOPMENT\b",
        r"\bINGENIERIA AMBIENTAL\b",
        r"\bTECNOLOGIA AMBIENTAL\b",
        r"\bCIENCIAS AMBIENTALES\b",
    ],
    "Inteligencia Artificial": [
        # Términos generales y políglotas
        r"\bINTELIGENCIA ARTIFI",
        r"\bARTIFICIAL INTELLIGEN",
        r"\bINTELLIGENCE ARTIFI",
        r"\bKUNSTLICHE INTELLIGEN",
        r"\bIA GENERATIVA\b",
        r"\bIA APLICADA\b",
        # Fundamentos internacionales
        r"\bMACHINE LEARNING\b",
        r"\bAPRENDIZAJE AUTOMATICO\b",
        r"\bAPRENDIZAGEM AUTOMATICA\b",
        r"\bMASCHINELLES LERNEN\b",
        r"\bDEEP LEARNING\b",
        r"\bAPRENDIZAJE PROFUNDO\b",
        r"\bAPRENDIZAGEM PROFUNDA\b",
        r"\bPROCESAMIENTO DE LENGUAJE NATURAL\b",
        r"\bNATURAL LANGUAGE PROCESSING\b",
        r"\bPROCESSAMENTO DE LINGUAGEM NATURAL\b",
        r"\bTRAITEMENT AUTOMATIQUE DU LANGAGE\b",
        r"\bNLP\b",
        r"\bVISION POR COMPUTADOR\b",
        r"\bCOMPUTER VISION\b",
        r"\bVISION PAR ORDINATEUR\b",
        r"\bROBOTICA INTELIGENTE\b",
        r"\bINTELLIGENT ROBOTIC\b",
        r"\bDATA SCIENCE\b",
        r"\bCIENCIA DE DATOS\b",
        r"\bCIENCIA DE DADOS\b",
        # Gobernanza, Ética y Seguridad
        r"\bGOBERNANZA DE LA (INTELIGENCIA ARTIFICIAL|IA)\b",
        r"\bAI GOVERNANCE\b",
        r"\bETICA DE LA (INTELIGENCIA ARTIFICIAL|IA)\b",
        r"\bAI ETHICS\b",
        r"\bETHIQUE DE L IA\b",
        r"\bSEGURIDAD DE SISTEMAS DE IA\b",
        r"\bAI SAFETY\b",
        # Aplicaciones Verticales
        r"\bINTELIGENCIA ARTIFICIAL EN SALUD\b",
        r"\bAI IN HEALTHCARE\b",
        r"\bIA EN SALUD\b",
        r"\bIA MEDICA\b",
        r"\bMEDICAL AI\b",
        r"\bINTELIGENCIA ARTIFICIAL INDUSTRIAL\b",
        r"\bINDUSTRIAL AI\b",
        r"\bIA INDUSTRIAL\b",
        r"\bMANUFACTURA INTELIGENTE\b",
        r"\bSMART MANUFACTURING\b",
        r"\bAUTOMATIZACION INTELIGENTE\b",
        r"\bINTELLIGENT AUTOMATION\b",
        r"\bMANTENIMIENTO PREDICTIVO\b",
        r"\bPREDICTIVE MAINTENANCE\b",
        r"\bMAINTENANCE PREDICTIVE\b",
        r"\bINTELIGENCIA ARTIFICIAL AMBIENTAL\b",
        r"\bENVIRONMENTAL AI\b",
        r"\bIA AMBIENTAL\b",
        r"\bGESTION DE RIESGO DE DESASTRES\b",
        r"\bDISASTER RISK MANAGEMENT\b",
        r"\bMONITOREO AMBIENTAL INTELIGENTE\b",
        r"\bMODELAMIENTO PREDICTIVO\b",
        r"\bPREDICTIVE MODELING\b",
        r"\bALERTA TEMPRANA\b",
        r"\bEARLY WARNING\b",
    ],
    "CTIsalud": [
        # Salud general, ciencias de la salud y biomedicina
        r"\bSALUD",  # Atrapa SALUD, SALUD PUBLICA, SALUD MENTAL, etc.
        r"\bHEALTH",
        r"\bSANTE",
        r"\bSAUDE",
        r"\bGESUNDHEIT",
        r"\bBIOMEDICIN",
        r"\bINVESTIGACION CLINICA",
        # Fisiología, morfología y ciencias médicas básicas
        r"\bFISIOL",  # Atrapa FISIOLOGIA, FISIOPATOLOGIA, PHYSIOLOGY
        r"\bPHYSIOL",
        # Alimentación, nutrición y metabolismo vinculados a la salud
        r"\bALIMENTAC",  # Atrapa ALIMENTACION, ALIMENTAÇÃO
        r"\bNUTRICI",  # Atrapa NUTRICION, NUTRIÇAO
        r"\bNUTRIT",  # Atrapa NUTRITION
        r"\bMETABOL",
        r"\bSTOFFWECHSEL",
        r"\bOBESID",
        r"\bOBESIT",
        r"\bADIPOSI",
        # Biomarcadores y diagnóstico
        r"\bBIOMARCA",
        r"\bBIOMARKER",
        r"\bBIOMARQUEUR",
        r"\bDIAGNOSTI",
        r"\bFRUHERKENNUNG\b",
        # Antimicrobianos e infecciones
        r"\bANTIMICROB",
        r"\bANTIMICROBIE",
        r"\bANTIMIKROB",
        r"\bINFECCION",
        r"\bINFECCAO",
        r"\bINFECTION",
        r"\bINFEKTION",
        r"\bNOSOCOMI",
        r"\bHEALTHCARE ASSOCIATED INFECTION",
        r"\bHEALTHCARE ACQUIRED\b",
        # Epidemiología y neurociencias
        r"\bEPIDEMIO",
        r"\bNEUROBIOL",
        r"\bNEUROLOG",
        r"\bNEUROSCI",
        r"\bTRANSTORNO",
        r"\bTRASTORNO",
        r"\bDISORDER",
        r"\bTROUBLE MENTAL",
        r"\bPSYCHISCH",
        r"\bNERVENDRUCK",
        # Vectoriales y entomología
        r"\bMETAXENIC",
        r"\bVECTOR BORN",
        r"\bMALADIE A VECTEUR",
        r"\bDOENCAS TRANSMITIDAS POR VETORES\b",
        r"\bENTOMO",
        r"\bVECTOR CONTROL\b",
        r"\bCONTROLE DE VETORES\b",
        r"\bINSEKTENKONTROLLE\b",
        # Cáncer y oncología
        r"\bNEOPLAS",
        r"\bCANCER",
        r"\bCANCEROLOG",
        r"\bONCOL",
        r"\bKREBS",
        # Enfermedades no transmisibles y gestión de salud
        r"\bNON COMMUNICABLE",
        r"\bNON TRANSMISSIBLE",
        r"\bNAO TRANSMISSIVE",
        r"\bNICHTUBERTRAGBAR",
        r"\bGESTION DE SERVICIOS DE SALUD",
        r"\bHEALTH SERVICES MANAGEMENT",
        r"\bGESTION DES SERVICES DE SANTE",
        r"\bGESTÃO DE SERVIÇOS DE SAÚDE",
        r"\bGESUNDHEITSMANAGEMENT",
    ],
}


# 3. Función ejecutora de la clasificación
def categorizar_carrera_sintesis(carrera):
    carrera_norm = normalizar_texto(carrera)
    if not carrera_norm:
        return "Otros"

    for area, patrones in REGLAS_MATCHING_SINTESIS.items():
        for patron in patrones:
            if re.search(patron, carrera_norm):
                return area

    return "Otros"


# Aplicar al DataFrame
pronabec["AREA_PRIORIZADA"] = pronabec["CARRERA_PRONABEC"].apply(
    categorizar_carrera_sintesis)


pronabec["AREA_PRIORIZADA_BIN"] = pronabec["AREA_PRIORIZADA"].apply(
    lambda x: "NO" if x == "Otros" else "SI"
)

# Se analiza un caso para los registros que contienen biotecnología
caso = pronabec[pronabec["AREA_PRIORIZADA"]=="Biotecnología"]

# Se analiza un caso para registros que contienen el nombre de una carrera específica
caso = pronabec[pronabec["CARRERA_PRONABEC"]=="MSC CLIMATE CHANGE AND ARTIFICIAL INTELLIGENCE"]

###############################################################################
# ANALISIS DE NATURALEZA MACRO
###############################################################################

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
# Se analiza la evolución temporal de los programas de maestria
# asociados con las areas prioritarias
###############################################################################

# Configuración de estilo académico
sns.set_style("white")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"

# 1. Definición de columnas del dataset
col_anio = "AÑO_CONVOCATORIA"
col_target = "AREA_PRIORIZADA_BIN"  # Columna binaria ('SI' / 'NO')

# 2. Calcular porcentajes por año (Tabla cruzada)
df_cross = (
    pd.crosstab(
        becario_maestria[col_anio], becario_maestria[col_target], normalize="index"
    )
    * 100
)

# Extraer la serie temporal correspondiente a la categoría 'SI'
df_si = df_cross["SI"].reset_index()

# 3. Construcción del gráfico optimizado
fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

color_si = "#E65100"  # Naranja sobrio (Área Priorizada)
x_labels = df_si[col_anio].astype(str)
x_pos = np.arange(len(x_labels))

# Trazar la línea de tendencia y área bajo la curva
ax.plot(
    x_pos,
    df_si["SI"],
    color=color_si,
    linewidth=2.5,
    marker="o",
    markersize=7,
    label="Área Priorizada (SI)",
)
ax.fill_between(x_pos, df_si["SI"], color=color_si, alpha=0.12)

# 4. Anotación inteligente de valores sobre cada punto de la línea
for i, val in enumerate(df_si["SI"]):
    ax.annotate(
        f"{val:.1f}%",
        (x_pos[i], val),
        textcoords="offset points",
        xytext=(0, 8),
        ha="center",
        fontsize=8.5,
        fontweight="bold",
        color="#212121",
    )

# 5. Estética y escala optimizada
max_val = df_si["SI"].max()
ax.set_ylim(0, min(100, max_val * 1.15))

ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels, fontsize=9.5)
ax.set_ylabel("Participación (%)", fontsize=10, labelpad=10)
ax.set_xlabel("Año de Convocatoria", fontsize=10, labelpad=10)

#ax.set_title(
    #"Evolución de la Cobertura en Áreas Prioritarias (2013-2025)",
    #fontsize=12,
    #fontweight="bold",
    #pad=15,
#)

# Grilla sutil y limpieza de bordes
ax.yaxis.grid(True, linestyle="--", alpha=0.4, color="#CCCCCC")
ax.set_axisbelow(True)
sns.despine(top=True, right=True)

plt.tight_layout()
plt.show()

###############################################################################
# Se analiza la tasa de participación en áreas priorizadas desagregada por rango
# de edad y género
###############################################################################

# 1. Crear la variable de rango etario
bins = [19, 29, 39, 120]
labels = ["20 - 29 años", "30 - 39 años", "40 - 50 años"]

becario_maestria["RANGO_EDAD"] = pd.cut(
    becario_maestria["EDADBASES"], bins=bins, labels=labels, right=True
)

# 2. CALCULO MACRO (ESTRUCTURAL / PARTE-TODO)
# Universo total de becarios de maestría (Denominador N)
N_total = len(becario_maestria)

# Filtrar únicamente los adjudicados en áreas priorizadas
priorizados = becario_maestria[becario_maestria["AREA_PRIORIZADA_BIN"] == "SI"]

# Calcular la participación relativa respecto al TOTAL GLOBAL (N_total)
df_plot = (
    priorizados.groupby(["RANGO_EDAD", "SEXO"], observed=False)
    .size()
    .reset_index(name="CONTEO")
)

df_plot["PARTICIPACION_MACRO"] = (df_plot["CONTEO"] / N_total) * 100

# Mostrar tabla cruzada en consola
tasa_macro = df_plot.pivot(
    index="RANGO_EDAD", columns="SEXO", values="PARTICIPACION_MACRO"
).fillna(0)
print("Participación Estructural sobre el Total de Maestría (%):")
print(tasa_macro.round(2))

# 3. CONFIGURACIÓN DEL GRÁFICO
sns.set_style("white")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"

fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)

palette_gender = {"MASCULINO": "#2B5C8F", "FEMENINO": "#D95F02"}

sns_bar = sns.barplot(
    data=df_plot,
    x="RANGO_EDAD",
    y="PARTICIPACION_MACRO",
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
            fontsize=11,
            fontweight="bold",
            color="#333333",
            xytext=(0, 4),
            textcoords="offset points",
        )

# Personalización de títulos y ejes
max_y = df_plot["PARTICIPACION_MACRO"].max()
ax.set_ylim(0, (max_y if not np.isnan(max_y) and max_y > 0 else 10) * 1.3)
ax.set_ylabel(
    "Participación sobre el Total de Becas (%)", fontsize=10, labelpad=10
)
ax.set_xlabel("Rango de edades", fontsize=10, labelpad=10)
ax.set_title(
    "Distribución Estructural de Becas en Áreas Priorizadas por Edad y Género",
    fontsize=12,
    fontweight="bold",
    pad=15,
)

# Grilla y acabado visual
ax.yaxis.grid(True, linestyle="--", alpha=0.3, color="#CCCCCC")
ax.set_axisbelow(True)
sns.despine(top=True, right=True)

# Leyenda
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
fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

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
    "Proporción del Total Anual (%)", fontsize=10, fontweight="bold", labelpad=10)
ax.set_xlabel("Año de Convocatoria", fontsize=10, fontweight="bold", labelpad=10)

#ax.set_title(
    #"Evolución de la Participación Relativa por Área Priorizada (2013-2025)",
    #fontsize=12,
    #fontweight="bold",
    #pad=15,
#)

# Ajuste dinámico del límite vertical para dar espacio a las anotaciones
max_y = df_plot.max().max()
ax.set_ylim(0, max_y * 1.25)

# Grilla y despinado
ax.yaxis.grid(True, linestyle="--", alpha=0.3, color="#CCCCCC")
ax.set_axisbelow(True)
sns.despine(top=True, right=True)

plt.legend(
    title='Área priorizada',
    loc='upper center',
    bbox_to_anchor=(0.5, -0.15),
    ncol=3,               # Distribuye los elementos en 3 columnas
    fontsize=8.5,
    title_fontsize=9.5,
    frameon=True
)

plt.tight_layout()
plt.show()

###############################################################################
# ANALISIS DE NATURALEZA MICRO
###############################################################################

# Para el análisis micro, se elimina la categoria otros
becario_maestria = becario_maestria[becario_maestria["AREA_PRIORIZADA"]!="Otros"]

# 2. Calcular Frecuencia Absoluta (Conteo) y Porcentaje
area_counts = becario_maestria["AREA_PRIORIZADA"].value_counts()
area_pcts = (
    becario_maestria["AREA_PRIORIZADA"]
    .value_counts(normalize=True)
    .round(4)
    * 100
)

# Unir en un solo DataFrame
area = pd.DataFrame({"Conteo": area_counts, "Porcentaje": area_pcts}).reset_index()
area.rename(columns={"index": "AREA_PRIORIZADA"}, inplace=True)

# 3. Configuración de estilo gráfico
sns.set_style("whitegrid")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.figure(figsize=(10, 6), dpi=300)

# 4. Creación del gráfico de barras horizontales
ax = sns.barplot(data=area, x="Porcentaje", y="AREA_PRIORIZADA", palette="magma")

# 5. Anotación inteligente y limpia
for i, row in area.iterrows():
    pct_val = row["Porcentaje"]
    count_val = int(row["Conteo"])

    # Umbral del 12% para decidir si el entero cabe adentro de la barra
    if pct_val >= 12.0:
        # Entero SOLO dentro de la barra en blanco
        ax.annotate(
            f"{count_val:,}",
            (pct_val * 0.85, i),
            ha="center",
            va="center",
            color="white",
            fontsize=13,
            fontweight="bold",
        )
        # Porcentaje afuera
        ax.annotate(
            f"{pct_val:.1f}%",
            (pct_val, i),
            ha="left",
            va="center",
            xytext=(7, 0),
            textcoords="offset points",
            color="#111111",
            fontsize=13,
            fontweight="bold",
        )
    else:
        # En barras pequeñas, ambos valores van afuera para no amontonar
        ax.annotate(
            f"{pct_val:.1f}% ({count_val:,})",
            (pct_val, i),
            ha="left",
            va="center",
            xytext=(7, 0),
            textcoords="offset points",
            color="#111111",
            fontsize=12,
            fontweight="bold",
        )

# 6. Formato de ejes y etiquetas
ax.tick_params(axis="x", labelsize=12)
ax.tick_params(axis="y", labelsize=13, labelcolor="#222222")
plt.xlabel("Porcentaje (%)", fontsize=14, fontweight="bold", labelpad=10)
plt.ylabel("Área Priorizada", fontsize=14, fontweight="bold", labelpad=10)

# Ajustar límite horizontal para dar margen a la etiqueta combinada
plt.xlim(0, max(area["Porcentaje"]) * 1.22)
sns.despine(top=True, right=True)

plt.tight_layout()
plt.show()

# Se calcula la edad mediana de los becarios de programas de maestria asociados con las areas prioriazadas
becario_maestria["EDADBASES"].describe()

# Se calcula la distribución entre hombres y mujeres
becario_maestria.SEXO.value_counts(normalize=True).round(2)*100

###############################################################################
# Se calcula un gráfico apilado para mostrar la participación de cada area priorizada
# entre ellas
###############################################################################
# 1. Definir paleta de colores idéntica
colores_areas = {
    "Ambiente y Cambio Climático": "#D32F2F",  # Rojo
    "CTIsalud": "#388E3C",  # Verde
    "Biotecnología": "#1976D2",  # Azul
    "Inteligencia Artificial": "#7B1FA2",  # Morado
    "Nanotecnología": "#F57C00",  # Naranja
}

# 2. Filtrar subconjunto (excluir "Otros" si existe en la base)
becarios_micro = becario_maestria[
    becario_maestria["AREA_PRIORIZADA"] != "Otros"
].copy()

# 3. Calcular frecuencias absolutas y totales por año (N_año) para mitigar el sesgo
df_counts = pd.crosstab(
    becarios_micro["AÑO_CONVOCATORIA"], becarios_micro["AREA_PRIORIZADA"]
)

totals_per_year = df_counts.sum(axis=1)

# Normalizar manualmente por fila para obtener porcentajes (%)
df_micro_cross = df_counts.div(totals_per_year, axis=0) * 100

# Reordenar columnas según la paleta definida
areas_ordenadas = [
    a for a in colores_areas.keys() if a in df_micro_cross.columns
]
df_micro_cross = df_micro_cross[areas_ordenadas]
df_counts = df_counts[areas_ordenadas]

# 4. Construcción del gráfico con muestra transparente
fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)

# Eje X transparente: Muestra el año y el número real de becarios de maestría
x_years_labels = [
    f"{year}\n(n={totals_per_year[year]})" for year in df_micro_cross.index
]
x_indices = np.arange(len(df_micro_cross))

bottom_val = np.zeros(len(df_micro_cross))

for area_name in areas_ordenadas:
    values_pct = df_micro_cross[area_name].values
    values_raw = df_counts[area_name].values
    color = colores_areas[area_name]

    bars = ax.bar(
        x_indices,
        values_pct,
        bottom=bottom_val,
        label=area_name,
        color=color,
        width=0.65,
        edgecolor="white",
        linewidth=0.8,
    )

    # Añadir porcentaje (y opcionalmente n) dentro del segmento si tiene suficiente altura (>= 6%)
    for i, val in enumerate(values_pct):
        if val >= 6.0:
            y_pos = bottom_val[i] + val / 2.0
            ax.annotate(
                f"{val:.0f}%",
                (x_indices[i], y_pos),
                ha="center",
                va="center",
                color="white",
                fontsize=11,
                fontweight="bold",
            )

    bottom_val += values_pct

# 5. Formato y estética institucional
ax.set_ylim(0, 100)
ax.set_xticks(x_indices)
ax.set_xticklabels(x_years_labels, fontsize=9.5)

ax.set_ylabel(
    "Distribución Relativa en Áreas Prioritarias (%)",
    fontsize=11,
    fontweight="bold",
    labelpad=10,
)
ax.set_xlabel(
    "Año de Convocatoria (Tamaño Muestral n)",
    fontsize=11,
    fontweight="bold",
    labelpad=10,
)

# Leyenda fuera del área de dibujo
ax.legend(
    title="Área Priorizada",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    frameon=True,
    facecolor="#F8F9FA",
    edgecolor="none",
    fontsize=10,
)

# Grilla limpia
ax.yaxis.grid(True, linestyle="--", alpha=0.3, color="#BBBBBB")
ax.set_axisbelow(True)
sns.despine(top=True, right=True)

plt.tight_layout()
plt.show()

##############################################################################
# Se calcula la composición de programas de maestria asociados con áreas
# priorizadas por Rango de Edad
###############################################################################
# 1. Definir la paleta exacta de colores homologada
colores_areas = {
    "Ambiente y Cambio Climático": "#D32F2F",  # Rojo
    "CTIsalud": "#388E3C",  # Verde
    "Biotecnología": "#1976D2",  # Azul
    "Inteligencia Artificial": "#7B1FA2",  # Morado
    "Nanotecnología": "#F57C00",  # Naranja
}

# 2. Obtener el universo total de la muestra (N)
N_total = len(becario_maestria)

# 3. Calcular frecuencias absolutas por Rango de Edad y Área Priorizada
df_grouped = (
    becario_maestria.groupby(["RANGO_EDAD", "AREA_PRIORIZADA"], observed=False)
    .size()
    .unstack(fill_value=0)
)

# Transformar a formato largo manteniendo conteos directos
df_plot = (
    df_grouped.reset_index()
    .melt(
        id_vars="RANGO_EDAD", var_name="AREA_PRIORIZADA", value_name="Conteo"
    )
)

# Calcular el porcentaje relativo sobre la muestra general N
df_plot["Pct_Total"] = (df_plot["Conteo"] / N_total) * 100

# 4. Configuración de estilo gráfico
sns.set_style("whitegrid")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

# 5. Creación del gráfico con frecuencias absolutas y paleta homologada
sns.barplot(
    data=df_plot,
    x="RANGO_EDAD",
    y="Conteo",
    hue="AREA_PRIORIZADA",
    palette=colores_areas,
    hue_order=[
        "Ambiente y Cambio Climático",
        "CTIsalud",
        "Biotecnología",
        "Inteligencia Artificial",
        "Nanotecnología",
    ],
    ax=ax,
)

# 6. Etiquetas dobles: Muestra el Conteo real (n) y el Porcentaje sobre el Total General (%)
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        pct_val = (height / N_total) * 100

        label_text = f"{int(height)}\n({pct_val:.1f}%)"

        ax.annotate(
            label_text,
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            xytext=(0, 7),  # Separación de 7 puntos respecto al tope de la barra
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
            color="#212121",
        )
# 7. Formato de títulos, ejes y escala Y dinámica
ax.set_xlabel("Rango de Edades", fontsize=12, fontweight="bold", labelpad=10)
ax.set_ylabel("Número de Becarios de maestría (n)", fontsize=12, fontweight="bold", labelpad=10)

# Margen superior para evitar desborde de etiquetas
max_height = df_plot["Conteo"].max()
ax.set_ylim(0, max_height * 1.25)

# Leyenda configurada
sns.move_legend(
    ax,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.35),  # Ubica la leyenda debajo del eje X
    ncol=3,  # Distribuye los 5 elementos en columnas horizontales
    title="Área Priorizada",
    title_fontsize=11,
    fontsize=9.5,
    frameon=True,
    facecolor="#FFFFFF",
    edgecolor="#CCCCCC",
)


sns.despine(top=True, right=True)
plt.tight_layout()
plt.show()

###############################################################################
# Se calcula la composición de programas de maestria asociados con áreas
# priorizadas por Género
###############################################################################
# 1. Definir la paleta exacta de colores homologada
colores_areas = {
    "Ambiente y Cambio Climático": "#D32F2F",  # Rojo
    "CTIsalud": "#388E3C",  # Verde
    "Biotecnología": "#1976D2",  # Azul
    "Inteligencia Artificial": "#7B1FA2",  # Morado
    "Nanotecnología": "#F57C00",  # Naranja
}

# 2. Filtrar subconjunto micro (excluir "Otros")
becario_micro = becario_maestria[
    becario_maestria["AREA_PRIORIZADA"] != "Otros"
].copy()

# Universo total analizado en este subconjunto (N)
N_total = len(becario_micro)

# 3. Calcular frecuencias absolutas en lugar de normalizar por índice
df_genero = (
    pd.crosstab(
        becario_micro["SEXO"],
        becario_micro["AREA_PRIORIZADA"],
    )
).reset_index()

# Transformar a formato largo para Seaborn
df_genero_melted = pd.melt(
    df_genero,
    id_vars=["SEXO"],
    var_name="AREA_PRIORIZADA",
    value_name="Conteo",
)

# 4. Estilo de figura
sns.set_style("whitegrid")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

# 5. Gráfico de barras basado en conteos absolutos
sns.barplot(
    data=df_genero_melted,
    x="SEXO",
    y="Conteo",
    hue="AREA_PRIORIZADA",
    palette=colores_areas,
    hue_order=[
        "Ambiente y Cambio Climático",
        "CTIsalud",
        "Biotecnología",
        "Inteligencia Artificial",
        "Nanotecnología",
    ],
    ax=ax,
)

# 6. Agregar etiquetas compactas en una sola línea con margen suficiente: "n (pct%)"
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        pct_val = (height / N_total) * 100

        # Texto en 2 líneas con salto \n
        label_text = f"{int(height)}\n({pct_val:.1f}%)"

        ax.annotate(
            label_text,
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            xytext=(0, 3),
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
            color="#212121",
            linespacing=0.85,  # Junta las dos líneas verticalmente
        )

# 7. Formato de ejes y límite superior
ax.set_xlabel("Género", fontsize=12, fontweight="bold", labelpad=10)
ax.set_ylabel(
    "Número de Becarios de maestría (n)", fontsize=12, fontweight="bold", labelpad=10
)

# Límite superior adaptativo para dar espacio a las anotaciones
max_height = df_genero_melted["Conteo"].max()
ax.set_ylim(0, max_height * 1.20)

# 8. Leyenda horizontal centrada en la parte inferior
sns.move_legend(
    ax,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.35),
    ncol=3,
    title="Área priorizada",
    title_fontsize=11,
    fontsize=9.5,
    frameon=True,
    facecolor="#FFFFFF",
    edgecolor="#CCCCCC",
)

sns.despine(top=True, right=True)
plt.tight_layout()
plt.show()


###############################################################################
# Se elabora un heatmap que muestra la interacción entre las areas priorizadas
# el rango de edades y el género
###############################################################################
# 1. Crear matriz de conteos absolutos
df_pivot = (
    becario_maestria.groupby(
        ["AREA_PRIORIZADA", "RANGO_EDAD", "SEXO"], observed=False
    )
    .size()
    .unstack(level=["RANGO_EDAD", "SEXO"], fill_value=0)
)

# Excluir "Otros" si corresponde
if "Otros" in df_pivot.index:
    df_pivot = df_pivot.drop(index="Otros")

# Reordenar columnas demográficas
column_order = [
    ("20 - 29 años", "FEMENINO"),
    ("20 - 29 años", "MASCULINO"),
    ("30 - 39 años", "FEMENINO"),
    ("30 - 39 años", "MASCULINO"),
    ("40 - 50 años", "FEMENINO"),
    ("40 - 50 años", "MASCULINO"),
]

column_order = [col for col in column_order if col in df_pivot.columns]
df_pivot = df_pivot[column_order]

# Calcular porcentaje sobre el TOTAL absoluto de la muestra (N)
N_total = df_pivot.values.sum()
heatmap_pct = (df_pivot / N_total) * 100

# 2. Crear matriz de texto combinada: "n\n(pct%)"
annot_matrix = df_pivot.copy().astype(str)
for col in df_pivot.columns:
    for idx in df_pivot.index:
        count = df_pivot.loc[idx, col]
        pct = heatmap_pct.loc[idx, col]
        if count > 0:
            annot_matrix.loc[idx, col] = f"{count}\n({pct:.1f}%)"
        else:
            annot_matrix.loc[idx, col] = "-"

# Nombres de etiquetas formateados con el total n por columna
col_totals = df_pivot.sum(axis=0)
formatted_columns = [
    f"{edad}\n({sexo[:3].upper()})\nn={col_totals[(edad, sexo)]}"
    for edad, sexo in column_order
]

# 3. Graficar Heatmap
plt.figure(figsize=(12, 6), dpi=300)
sns.set_style("white")

ax = sns.heatmap(
    heatmap_pct,  # Color según el % sobre el total de la muestra
    annot=annot_matrix,  # Muestra "n" y "(%)"
    fmt="",
    cmap="YlGnBu",
    linewidths=0.8,
    linecolor="white",
    cbar_kws={"label": "% del Total de Becarios", "shrink": 0.8},
    annot_kws={"size": 14, "weight": "bold"},
)

ax.set_xticklabels(formatted_columns, rotation=0, ha="center", fontsize=11)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)

plt.xlabel(
    "Cohorte Demográfica (Rango de Edad / Género)",
    fontsize=10.5,
    labelpad=12,
    fontweight="bold",
)

plt.ylabel("Área Priorizada", fontsize=10.5, labelpad=12, fontweight="bold")

plt.tight_layout()
plt.show()

###############################################################################
###############################################################################
# ANALISIS DE NATURALEZA MACRO PARA LOS DOCTORADO
###############################################################################
###############################################################################


###############################################################################
# Se obtiene un dataframe que almacena a los becarios de programas de doctorado
###############################################################################
becario_doctorado = adjudicado[adjudicado["NIVEL_EDUCATIVO"]=="DOCTORADO"]
becario_doctorado.columns

# Se analiza la participación de los programas de doctorado asociadas con las areas priorizadas
becario_doctorado.AREA_PRIORIZADA_BIN.value_counts(normalize=True).round(3)

###############################################################################
# Se analiza la evolución temporal de los programas de doctorado
# asociados con las areas prioritarias
###############################################################################
# 1. Configuración de variables para Doctorado
col_anio = "AÑO_CONVOCATORIA"
col_target = "AREA_PRIORIZADA_BIN"  # 'SI' / 'NO'

# Frecuencias absolutas y totales por año en el subconjunto becario_doctorado
df_counts_doc = pd.crosstab(
    becario_doctorado[col_anio], becario_doctorado[col_target]
)
totals_per_year_doc = df_counts_doc.sum(axis=1)

# Porcentajes normalizados a nivel Macro (Doctorado)
df_cross_doc = df_counts_doc.div(totals_per_year_doc, axis=0) * 100
df_si_doc = df_cross_doc["SI"].reset_index()

# 2. Construcción del gráfico Macro de Doctorado
fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)

color_si = "#E65100"  # Naranja sobrio
x_pos = np.arange(len(df_si_doc))

# Etiquetas del eje X con el tamaño total de la cohorte anual de doctorado
x_labels = [
    f"{year}\n(N={totals_per_year_doc[year]})" for year in df_si_doc[col_anio]
]

# Línea de tendencia Macro
ax.plot(
    x_pos,
    df_si_doc["SI"],
    color=color_si,
    linewidth=2.5,
    marker="o",
    markersize=7,
    label="Área Priorizada (SI)",
)
ax.fill_between(x_pos, df_si_doc["SI"], color=color_si, alpha=0.12)

# 3. Anotaciones dobles: Porcentaje % y Número de becarios n
for i, val in enumerate(df_si_doc["SI"]):
    year_val = df_si_doc[col_anio].iloc[i]
    n_si = df_counts_doc.loc[year_val, "SI"]

    # Despliegue de etiqueta con n absoluto
    ax.annotate(
        f"{val:.1f}%\n(n={n_si})",
        (x_pos[i], val),
        textcoords="offset points",
        xytext=(0, 8),
        ha="center",
        fontsize=8.5,
        fontweight="bold",
        color="#212121",
        linespacing=0.85,
    )

# 4. Formato de ejes y limites
max_val = df_si_doc["SI"].max()
ax.set_ylim(0, min(100, max_val * 1.25))

ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels, fontsize=9)

ax.set_ylabel(
    "Cobertura en Áreas Prioritarias (en % de Doctorado)",
    fontsize=10,
    fontweight="bold",
    labelpad=10,
)
ax.set_xlabel(
    "Año de Convocatoria (Población Total de Doctorado N)",
    fontsize=10,
    fontweight="bold",
    labelpad=10,
)

ax.yaxis.grid(True, linestyle="--", alpha=0.4, color="#CCCCCC")
ax.set_axisbelow(True)
sns.despine(top=True, right=True)

plt.tight_layout()
plt.show()

###############################################################################
# Se analiza la tasa de participación en áreas priorizadas desagregada por rango
# de edad y género
###############################################################################

# 1. Crear la variable de rango etario
bins = [19, 29, 39, 120]
labels = ["20 - 29 años", "30 - 39 años", "40 - 50 años"]

becario_doctorado["RANGO_EDAD"] = pd.cut(
    becario_doctorado["EDADBASES"], bins=bins, labels=labels, right=True
)

# 2. CALCULO MACRO (ESTRUCTURAL / PARTE-TODO)
# Universo total de becarios de maestría (Denominador N)
N_total = len(becario_maestria)

# Filtrar únicamente los adjudicados en áreas priorizadas
priorizados = becario_doctorado[becario_doctorado["AREA_PRIORIZADA_BIN"] == "SI"]

# Calcular la participación relativa respecto al TOTAL GLOBAL (N_total)
df_plot = (
    priorizados.groupby(["RANGO_EDAD", "SEXO"], observed=False)
    .size()
    .reset_index(name="CONTEO")
)

df_plot["PARTICIPACION_MACRO"] = (df_plot["CONTEO"] / N_total) * 100

# Mostrar tabla cruzada en consola
tasa_macro = df_plot.pivot(
    index="RANGO_EDAD", columns="SEXO", values="PARTICIPACION_MACRO"
).fillna(0)
print("Participación Estructural sobre el Total de Maestría (%):")
print(tasa_macro.round(2))

# 3. CONFIGURACIÓN DEL GRÁFICO
sns.set_style("white")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"

fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)

palette_gender = {"MASCULINO": "#2B5C8F", "FEMENINO": "#D95F02"}

sns_bar = sns.barplot(
    data=df_plot,
    x="RANGO_EDAD",
    y="PARTICIPACION_MACRO",
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
            fontsize=11,
            fontweight="bold",
            color="#333333",
            xytext=(0, 4),
            textcoords="offset points",
        )

# Personalización de títulos y ejes
max_y = df_plot["PARTICIPACION_MACRO"].max()
ax.set_ylim(0, (max_y if not np.isnan(max_y) and max_y > 0 else 10) * 1.3)
ax.set_ylabel(
    "Participación sobre el Total de Becas (%)", fontsize=10, labelpad=10
)
ax.set_xlabel("Rango de edades", fontsize=10, labelpad=10)
ax.set_title(
    "Distribución Estructural de Becas en Áreas Priorizadas por Edad y Género",
    fontsize=12,
    fontweight="bold",
    pad=15,
)

# Grilla y acabado visual
ax.yaxis.grid(True, linestyle="--", alpha=0.3, color="#CCCCCC")
ax.set_axisbelow(True)
sns.despine(top=True, right=True)

# Leyenda
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
# Se analiza la evolución temporal de cada área priorizada con respecto al total
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
        becario_doctorado[col_anio], becario_doctorado[col_area], normalize="index"
    )
    * 100
)

# 2. Filtrar únicamente las áreas priorizadas específicas (excluyendo 'Otros' o 'No Priorizado')
areas_especificas = [
    col for col in df_relativo.columns if col not in ["Otros"]
]
df_plot = df_relativo[areas_especificas]

# 3. Construir el gráfico de líneas de evolución
fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

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
    "Proporción del Total Anual (%)", fontsize=10, fontweight="bold", labelpad=10)
ax.set_xlabel("Año de Convocatoria", fontsize=10, fontweight="bold", labelpad=10)

#ax.set_title(
    #"Evolución de la Participación Relativa por Área Priorizada (2013-2025)",
    #fontsize=12,
    #fontweight="bold",
    #pad=15,
#)

# Ajuste dinámico del límite vertical para dar espacio a las anotaciones
max_y = df_plot.max().max()
ax.set_ylim(0, max_y * 1.25)

# Grilla y despinado
ax.yaxis.grid(True, linestyle="--", alpha=0.3, color="#CCCCCC")
ax.set_axisbelow(True)
sns.despine(top=True, right=True)

plt.legend(
    title='Área priorizada',
    loc='upper center',
    bbox_to_anchor=(0.5, -0.15),
    ncol=3,               # Distribuye los elementos en 3 columnas
    fontsize=8.5,
    title_fontsize=9.5,
    frameon=True
)

plt.tight_layout()
plt.show()

###############################################################################
# ANALISIS DE NATURALEZA MICRO
###############################################################################

# Para el análisis micro, se elimina la categoria otros
becario_doctorado = becario_doctorado[becario_doctorado["AREA_PRIORIZADA"]!="Otros"]

# 2. Calcular Frecuencia Absoluta (Conteo) y Porcentaje
area_counts = becario_doctorado["AREA_PRIORIZADA"].value_counts()
area_pcts = (
    becario_doctorado["AREA_PRIORIZADA"]
    .value_counts(normalize=True)
    .round(4)
    * 100
)

# Unir en un solo DataFrame
area = pd.DataFrame({"Conteo": area_counts, "Porcentaje": area_pcts}).reset_index()
area.rename(columns={"index": "AREA_PRIORIZADA"}, inplace=True)

# 3. Configuración de estilo gráfico
sns.set_style("whitegrid")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.figure(figsize=(10, 6), dpi=300)

# 4. Creación del gráfico de barras horizontales
ax = sns.barplot(data=area, x="Porcentaje", y="AREA_PRIORIZADA", palette="magma")

# 5. Anotación inteligente y limpia
for i, row in area.iterrows():
    pct_val = row["Porcentaje"]
    count_val = int(row["Conteo"])

    # Umbral del 12% para decidir si el entero cabe adentro de la barra
    if pct_val >= 12.0:
        # Entero SOLO dentro de la barra en blanco
        ax.annotate(
            f"{count_val:,}",
            (pct_val * 0.85, i),
            ha="center",
            va="center",
            color="white",
            fontsize=13,
            fontweight="bold",
        )
        # Porcentaje afuera
        ax.annotate(
            f"{pct_val:.1f}%",
            (pct_val, i),
            ha="left",
            va="center",
            xytext=(7, 0),
            textcoords="offset points",
            color="#111111",
            fontsize=13,
            fontweight="bold",
        )
    else:
        # En barras pequeñas, ambos valores van afuera para no amontonar
        ax.annotate(
            f"{pct_val:.1f}% ({count_val:,})",
            (pct_val, i),
            ha="left",
            va="center",
            xytext=(7, 0),
            textcoords="offset points",
            color="#111111",
            fontsize=12,
            fontweight="bold",
        )

# 6. Formato de ejes y etiquetas
ax.tick_params(axis="x", labelsize=12)
ax.tick_params(axis="y", labelsize=13, labelcolor="#222222")
plt.xlabel("Porcentaje (%)", fontsize=14, fontweight="bold", labelpad=10)
plt.ylabel("Área Priorizada", fontsize=14, fontweight="bold", labelpad=10)

# Ajustar límite horizontal para dar margen a la etiqueta combinada
plt.xlim(0, max(area["Porcentaje"]) * 1.22)
sns.despine(top=True, right=True)

plt.tight_layout()
plt.show()

# Se calcula la edad mediana de los becarios de programas de maestria asociados con las areas prioriazadas
becario_doctorado["EDADBASES"].describe()

# Se calcula la distribución entre hombres y mujeres
becario_doctorado.SEXO.value_counts(normalize=True).round(2)*100

###############################################################################
# Se calcula un gráfico apilado para mostrar la participación de cada area priorizada
# entre ellas
###############################################################################
# 1. Definir paleta de colores idéntica
colores_areas = {
    "Ambiente y Cambio Climático": "#D32F2F",  # Rojo
    "CTIsalud": "#388E3C",  # Verde
    "Biotecnología": "#1976D2",  # Azul
    "Inteligencia Artificial": "#7B1FA2",  # Morado
    "Nanotecnología": "#F57C00",  # Naranja
}

# 2. Filtrar subconjunto (excluir "Otros" si existe en la base)
becarios_micro = becario_doctorado[
    becario_doctorado["AREA_PRIORIZADA"] != "Otros"
].copy()

# 3. Calcular frecuencias absolutas y totales por año (N_año) para mitigar el sesgo
df_counts = pd.crosstab(
    becarios_micro["AÑO_CONVOCATORIA"], becarios_micro["AREA_PRIORIZADA"]
)

totals_per_year = df_counts.sum(axis=1)

# Normalizar manualmente por fila para obtener porcentajes (%)
df_micro_cross = df_counts.div(totals_per_year, axis=0) * 100

# Reordenar columnas según la paleta definida
areas_ordenadas = [
    a for a in colores_areas.keys() if a in df_micro_cross.columns
]
df_micro_cross = df_micro_cross[areas_ordenadas]
df_counts = df_counts[areas_ordenadas]

# 4. Construcción del gráfico con muestra transparente
fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)

# Eje X transparente: Muestra el año y el número real de becarios de maestría
x_years_labels = [
    f"{year}\n(n={totals_per_year[year]})" for year in df_micro_cross.index
]
x_indices = np.arange(len(df_micro_cross))

bottom_val = np.zeros(len(df_micro_cross))

for area_name in areas_ordenadas:
    values_pct = df_micro_cross[area_name].values
    values_raw = df_counts[area_name].values
    color = colores_areas[area_name]

    bars = ax.bar(
        x_indices,
        values_pct,
        bottom=bottom_val,
        label=area_name,
        color=color,
        width=0.65,
        edgecolor="white",
        linewidth=0.8,
    )

    # Añadir porcentaje (y opcionalmente n) dentro del segmento si tiene suficiente altura (>= 6%)
    for i, val in enumerate(values_pct):
        if val >= 6.0:
            y_pos = bottom_val[i] + val / 2.0
            ax.annotate(
                f"{val:.0f}%",
                (x_indices[i], y_pos),
                ha="center",
                va="center",
                color="white",
                fontsize=11,
                fontweight="bold",
            )

    bottom_val += values_pct

# 5. Formato y estética institucional
ax.set_ylim(0, 100)
ax.set_xticks(x_indices)
ax.set_xticklabels(x_years_labels, fontsize=9.5)

ax.set_ylabel(
    "Distribución Relativa en Áreas Prioritarias (%)",
    fontsize=11,
    fontweight="bold",
    labelpad=10,
)
ax.set_xlabel(
    "Año de Convocatoria (Tamaño Muestral n)",
    fontsize=11,
    fontweight="bold",
    labelpad=10,
)

# Leyenda fuera del área de dibujo
ax.legend(
    title="Área Priorizada",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    frameon=True,
    facecolor="#F8F9FA",
    edgecolor="none",
    fontsize=10,
)

# Grilla limpia
ax.yaxis.grid(True, linestyle="--", alpha=0.3, color="#BBBBBB")
ax.set_axisbelow(True)
sns.despine(top=True, right=True)

plt.tight_layout()
plt.show()

##############################################################################
# Se calcula la composición de programas de maestria asociados con áreas
# priorizadas por Rango de Edad
###############################################################################
# 1. Definir la paleta exacta de colores homologada
colores_areas = {
    "Ambiente y Cambio Climático": "#D32F2F",  # Rojo
    "CTIsalud": "#388E3C",  # Verde
    "Biotecnología": "#1976D2",  # Azul
    "Inteligencia Artificial": "#7B1FA2",  # Morado
    "Nanotecnología": "#F57C00",  # Naranja
}

# 2. Obtener el universo total de la muestra (N)
N_total = len(becario_doctorado)

# 3. Calcular frecuencias absolutas por Rango de Edad y Área Priorizada
df_grouped = (
    becario_doctorado.groupby(["RANGO_EDAD", "AREA_PRIORIZADA"], observed=False)
    .size()
    .unstack(fill_value=0)
)

# Transformar a formato largo manteniendo conteos directos
df_plot = (
    df_grouped.reset_index()
    .melt(
        id_vars="RANGO_EDAD", var_name="AREA_PRIORIZADA", value_name="Conteo"
    )
)

# Calcular el porcentaje relativo sobre la muestra general N
df_plot["Pct_Total"] = (df_plot["Conteo"] / N_total) * 100

# 4. Configuración de estilo gráfico
sns.set_style("whitegrid")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

# 5. Creación del gráfico con frecuencias absolutas y paleta homologada
sns.barplot(
    data=df_plot,
    x="RANGO_EDAD",
    y="Conteo",
    hue="AREA_PRIORIZADA",
    palette=colores_areas,
    hue_order=[
        "Ambiente y Cambio Climático",
        "CTIsalud",
        "Biotecnología",
        "Inteligencia Artificial",
        "Nanotecnología",
    ],
    ax=ax,
)

# 6. Etiquetas dobles: Muestra el Conteo real (n) y el Porcentaje sobre el Total General (%)
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        pct_val = (height / N_total) * 100

        label_text = f"{int(height)}\n({pct_val:.1f}%)"

        ax.annotate(
            label_text,
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            xytext=(0, 7),  # Separación de 7 puntos respecto al tope de la barra
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
            color="#212121",
        )
# 7. Formato de títulos, ejes y escala Y dinámica
ax.set_xlabel("Rango de Edades", fontsize=12, fontweight="bold", labelpad=10)
ax.set_ylabel("Número de Becarios de maestría (n)", fontsize=12, fontweight="bold", labelpad=10)

# Margen superior para evitar desborde de etiquetas
max_height = df_plot["Conteo"].max()
ax.set_ylim(0, max_height * 1.25)

# Leyenda configurada
sns.move_legend(
    ax,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.35),  # Ubica la leyenda debajo del eje X
    ncol=3,  # Distribuye los 5 elementos en columnas horizontales
    title="Área Priorizada",
    title_fontsize=11,
    fontsize=9.5,
    frameon=True,
    facecolor="#FFFFFF",
    edgecolor="#CCCCCC",
)


sns.despine(top=True, right=True)
plt.tight_layout()
plt.show()

###############################################################################
# Se calcula la composición de programas de maestria asociados con áreas
# priorizadas por Género
###############################################################################
# 1. Definir la paleta exacta de colores homologada
colores_areas = {
    "Ambiente y Cambio Climático": "#D32F2F",  # Rojo
    "CTIsalud": "#388E3C",  # Verde
    "Biotecnología": "#1976D2",  # Azul
    "Inteligencia Artificial": "#7B1FA2",  # Morado
    "Nanotecnología": "#F57C00",  # Naranja
}

# 2. Filtrar subconjunto micro (excluir "Otros")
becario_micro = becario_doctorado[
    becario_doctorado["AREA_PRIORIZADA"] != "Otros"
].copy()

# Universo total analizado en este subconjunto (N)
N_total = len(becario_micro)

# 3. Calcular frecuencias absolutas en lugar de normalizar por índice
df_genero = (
    pd.crosstab(
        becario_micro["SEXO"],
        becario_micro["AREA_PRIORIZADA"],
    )
).reset_index()

# Transformar a formato largo para Seaborn
df_genero_melted = pd.melt(
    df_genero,
    id_vars=["SEXO"],
    var_name="AREA_PRIORIZADA",
    value_name="Conteo",
)

# 4. Estilo de figura
sns.set_style("whitegrid")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

# 5. Gráfico de barras basado en conteos absolutos
sns.barplot(
    data=df_genero_melted,
    x="SEXO",
    y="Conteo",
    hue="AREA_PRIORIZADA",
    palette=colores_areas,
    hue_order=[
        "Ambiente y Cambio Climático",
        "CTIsalud",
        "Biotecnología",
        "Inteligencia Artificial",
        "Nanotecnología",
    ],
    ax=ax,
)

# 6. Agregar etiquetas compactas en una sola línea con margen suficiente: "n (pct%)"
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        pct_val = (height / N_total) * 100

        # Texto en 2 líneas con salto \n
        label_text = f"{int(height)}\n({pct_val:.1f}%)"

        ax.annotate(
            label_text,
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            xytext=(0, 3),
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
            color="#212121",
            linespacing=0.85,  # Junta las dos líneas verticalmente
        )

# 7. Formato de ejes y límite superior
ax.set_xlabel("Género", fontsize=12, fontweight="bold", labelpad=10)
ax.set_ylabel(
    "Número de Becarios de maestría (n)", fontsize=12, fontweight="bold", labelpad=10
)

# Límite superior adaptativo para dar espacio a las anotaciones
max_height = df_genero_melted["Conteo"].max()
ax.set_ylim(0, max_height * 1.20)

# 8. Leyenda horizontal centrada en la parte inferior
sns.move_legend(
    ax,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.35),
    ncol=3,
    title="Área priorizada",
    title_fontsize=11,
    fontsize=9.5,
    frameon=True,
    facecolor="#FFFFFF",
    edgecolor="#CCCCCC",
)

sns.despine(top=True, right=True)
plt.tight_layout()
plt.show()


###############################################################################
# Se elabora un heatmap que muestra la interacción entre las areas priorizadas
# el rango de edades y el género
###############################################################################
# 1. Crear matriz de conteos absolutos
df_pivot = (
    becario_doctorado.groupby(
        ["AREA_PRIORIZADA", "RANGO_EDAD", "SEXO"], observed=False
    )
    .size()
    .unstack(level=["RANGO_EDAD", "SEXO"], fill_value=0)
)

# Excluir "Otros" si corresponde
if "Otros" in df_pivot.index:
    df_pivot = df_pivot.drop(index="Otros")

# Reordenar columnas demográficas
column_order = [
    ("20 - 29 años", "FEMENINO"),
    ("20 - 29 años", "MASCULINO"),
    ("30 - 39 años", "FEMENINO"),
    ("30 - 39 años", "MASCULINO"),
    ("40 - 50 años", "FEMENINO"),
    ("40 - 50 años", "MASCULINO"),
]

column_order = [col for col in column_order if col in df_pivot.columns]
df_pivot = df_pivot[column_order]

# Calcular porcentaje sobre el TOTAL absoluto de la muestra (N)
N_total = df_pivot.values.sum()
heatmap_pct = (df_pivot / N_total) * 100

# 2. Crear matriz de texto combinada: "n\n(pct%)"
annot_matrix = df_pivot.copy().astype(str)
for col in df_pivot.columns:
    for idx in df_pivot.index:
        count = df_pivot.loc[idx, col]
        pct = heatmap_pct.loc[idx, col]
        if count > 0:
            annot_matrix.loc[idx, col] = f"{count}\n({pct:.1f}%)"
        else:
            annot_matrix.loc[idx, col] = "-"

# Nombres de etiquetas formateados con el total n por columna
col_totals = df_pivot.sum(axis=0)
formatted_columns = [
    f"{edad}\n({sexo[:3].upper()})\nn={col_totals[(edad, sexo)]}"
    for edad, sexo in column_order
]

# 3. Graficar Heatmap
plt.figure(figsize=(12, 6), dpi=300)
sns.set_style("white")

ax = sns.heatmap(
    heatmap_pct,  # Color según el % sobre el total de la muestra
    annot=annot_matrix,  # Muestra "n" y "(%)"
    fmt="",
    cmap="YlGnBu",
    linewidths=0.8,
    linecolor="white",
    cbar_kws={"label": "% del Total de Becarios", "shrink": 0.8},
    annot_kws={"size": 14, "weight": "bold"},
)

ax.set_xticklabels(formatted_columns, rotation=0, ha="center", fontsize=11)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)

plt.xlabel(
    "Cohorte Demográfica (Rango de Edad / Género)",
    fontsize=10.5,
    labelpad=12,
    fontweight="bold",
)

plt.ylabel("Área Priorizada", fontsize=10.5, labelpad=12, fontweight="bold")

plt.tight_layout()
plt.show()