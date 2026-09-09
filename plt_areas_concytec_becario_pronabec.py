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
# 1. Función para normalizar texto (remueve tildes y pasa a mayúsculas)
def normalizar_texto(texto):
    if not isinstance(texto, str):
        return ""
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join([c for c in texto if unicodedata.category(c) != "Mn"])
    return texto.upper().strip()


# 2. Diccionario con ORDEN DE PRECEDENCIA (Específico -> General)
REGLAS_MATCHING = {
    "Biotecnología": [
        # Frases específicas y compuestas PRIMERO
        r"\bBIOTECNOLOGIA BIODIVERSA\b",
        r"\bBIOTECNOLOGIA INDUSTRIAL\b",
        r"\bBIOENERGIA\b",
        r"\bBIORREMEDIACION\b",
        r"\bBIOTECNOLOGIA SANITARIA\b",
        r"\bBIOTECNOLOGIA AGROPECUARIA\b",
        r"\bBIOTECNOLOGIA PESQUERA\b",
        r"\bBIOTECNOLOGIA ACUICOLA\b",
        r"\bTECNOLOGIA AGROALIMENTARIA\b"
        # Raíces generales / comodines AL FINAL
        r"\bBIOTECNO",  # Captura Biotecnología, Biotecnológica, etc.
        r"\bBIOTECH"
    ],
    
    
    "Nanotecnología": [
        # Frases compuestas o específicas si deseas granularidad
        r"\bMATERIALES AVANZADOS\b",
        # Raíces amplias (comodines) que capturan plurales y variaciones
        r"\bNANOTEC",  # Captura NANOTECNOLOGÍA, NANOTECNOLOGÍAS, NANOTECNOLÓGICO, NANOTECH
        r"\bNANOCIEN",  # Captura NANOCIENCIA, NANOCIENCIAS
        r"\bNANOMAT",  # Captura NANOMATERIAL, NANOMATERIALES
        r"\bNANOESTR",  # Captura NANOESTRUCTURAS, NANOESTRUCTURADO
    ],
    
    
   "Ambiente y Cambio Climático": [
        # 1. Frases compuestas específicas (EVALUAR PRIMERO)
        r"\bCAMBIO CLIMATICO\b",
        r"\bTRANSICION ENERGETICA\b",
        r"\bTECNOLOGIAS ECOEFICIENTES\b",
        r"\bSISTEMAS ENERGETICOS INTELIGENTES\b",
        r"\bRESILIENCIA CLIMATICA\b",
        r"\bDIGITALIZACION ENERGETICA\b",
        # 2. Raíces y combinaciones temáticas clave
        r"\bDESARROLLO SOSTENIBLE\b",
        r"\bCIENCIAS AMBIENTALES\b",
        r"\bINGENIERIA AMBIENTAL\b",
        r"\bTECNOLOGIA AMBIENTAL\b",
        r"\bDESCARBONIZACION\b",
        r"\bECOEFICIEN",  # Captura Ecoeficiente, Ecoeficiencia, Ecoeficientes
        r"\bENERGIA RENOVABLE",  # Captura Energía Renovable, Energías Renovables
        r"\bENERGIA LIMPIA",  # Captura Energía Limpia, Energías Limpias
        r"\bEFICIENCIA ENERGETICA\b",
    ],

   "Inteligencia Artificial": [
        # 1. Concepto general, siglas y equivalentes políglotas
        r"\bINTELIGENCIA ARTIFI",    # ES: Inteligencia Artificial / Artificiales
        r"\bARTIFICIAL INTELLIGEN", # EN: Artificial Intelligence
        r"\bINTELLIGENCE ARTIFI",   # FR: Intelligence Artificielle / Artificielles
        r"\bINTELIGENCIA ARTIFI",   # PT: Inteligência Artificial (al normalizar coincide con ES)
        r"\bKUNSTLICHE INTELLIGEN", # DE: Künstliche Intelligenz (sin tildes/ümLAUT)
        r"\bIA GENERATIVA\b",
        r"\bIA APLICADA\b",
        # 2. Fundamentos y subcampos principales (Internacionales)
        r"\bMACHINE LEARNING\b",
        r"\bAPRENDIZAJE AUTOMATICO\b",
        r"\bDEEP LEARNING\b",
        r"\bAPRENDIZAJE PROFUNDO\b",
        r"\bPROCESAMIENTO DE LENGUAJE NATURAL\b",
        r"\bNATURAL LANGUAGE PROCESSING\b",  # EN
        r"\bTRAITEMENT AUTOMATIQUE DU LANGAGE\b", # FR (TAL)
        r"\bPROCESSAMENTO DE LINGUAGEM NATURAL\b", # PT
        r"\bMASCHINELLES LERNEN\b", # DE
        r"\bNLP\b",
        r"\bVISION POR COMPUTADOR\b",
        r"\bCOMPUTER VISION\b",      # EN
        r"\bVISION PAR ORDINATEUR\b",# FR
        r"\bROBOTICA INTELIGENTE\b",
        r"\bINTELLIGENT ROBOTIC\b", # EN/FR
        
        # 3. Gobernanza, Ética y Seguridad
        r"\bGOBERNANZA DE LA (INTELIGENCIA ARTIFICIAL|IA)\b",
        r"\bAI GOVERNANCE\b",        # EN
        r"\bETICA DE LA (INTELIGENCIA ARTIFICIAL|IA)\b",
        r"\bAI ETHICS\b",            # EN
        r"\bETHIQUE DE L'IA\b",      # FR
        r"\bSEGURIDAD DE SISTEMAS DE IA\b",
        r"\bAI SAFETY\b",            # EN
        
        # 4. Verticales Aplicadas (Salud, Industria, Ambiente, Riesgos)
        r"\bINTELIGENCIA ARTIFICIAL EN SALUD\b",
        r"\bAI IN HEALTHCARE\b",     # EN
        r"\bIA EN SALUD\b",
        r"\bIA MEDICA\b",
        r"\bMEDICAL AI\b",           # EN
        r"\bINTELIGENCIA ARTIFICIAL INDUSTRIAL\b",
        r"\bINDUSTRIAL AI\b",        # EN
        r"\bIA INDUSTRIAL\b",
        r"\bMANUFACTURA INTELIGENTE\b",
        r"\bSMART MANUFACTURING\b",  # EN
        r"\bAUTOMATIZACION INTELIGENTE\b",
        r"\bINTELLIGENT AUTOMATION\b", # EN
        r"\bMANTENIMIENTO PREDICTIVO\b",
        r"\bPREDICTIVE MAINTENANCE\b", # EN
        r"\bMAINTENANCE PREDICTIVE\b", # FR
        r"\bINTELIGENCIA ARTIFICIAL AMBIENTAL\b",
        r"\bENVIRONMENTAL AI\b",    # EN
        r"\bIA AMBIENTAL\b",
        r"\bGESTION DE RIESGO DE DESASTRES\b",
        r"\bDISASTER RISK MANAGEMENT\b", # EN
        r"\bMONITOREO AMBIENTAL INTELIGENTE\b",
        r"\bMODELAMIENTO PREDICTIVO\b",
        r"\bPREDICTIVE MODELING\b",  # EN
        r"\bALERTA TEMPRANA\b",
        r"\bEARLY WARNING\b",        # EN
        ],  

    "CTIsalud": [
    # 1. BIOMARCADORES Y DIAGNÓSTICO TEMPRANO
    # ES / EN / FR / PT / DE (Biomarkers, Biomarcadores, Biomarqueurs, Biomarker, Early/Precoz/Precoce/Früh)
        r"\bBIOMARCA",
        r"\bBIOMARKER",
        r"\bBIOMARQUEUR",
        r"\bDIAGNOSTI",  # Captura Diagnóstico, Diagnostic, Diagnóstico (ES, EN, FR, PT)
        r"\bFRUHERKENNUNG\b",  # DE: Diagnóstico temprano
        # 2. USO DE ANTIMICROBIANOS E INFECCIONES INTRAHOSPITALARIAS (IAAS / HAI)
        # ES / EN / FR / PT / DE
        r"\bANTIMICROB",  # ES, EN, PT: Antimicrobianos
        r"\bANTIMICROBIE",  # FR: Antimicrobiens
        r"\bANTIMIKROB",  # DE: Antimikrobiell
        r"\bINFECCION",  # ES: Infecciones
        r"\bINFECCAO",  # PT: Infecção
        r"\bINFECTION",  # EN, FR: Infection
        r"\bINFEKTION",  # DE: Infektion
        r"\bNOSOCOMI",  # Transversal: Infecciones nosocomiales
        r"\bHEALTHCARE ASSOCIATED INFECTION",  # EN: HAI
        r"\bHEALTHCARE ACQUIRED\b",
        # 3. EPIDEMIOLOGÍA, NEUROBIOLOGÍA, TRASTORNOS MENTALES Y SISTEMA NERVIOSO
        # ES / EN / FR / PT / DE
        r"\bEPIDEMIO",  # Transversal: Epidemiología / Epidemiology / Épidémiologie / Epidemiologia / Epidemiologie
        r"\bNEUROBIOL",  # Transversal: Neurobiología
        r"\bNEUROLOG",  # Transversal: Neurología
        r"\bNEUROSCI",  # EN, FR: Neurosciences
        r"\bMENTAL",  # ES, EN, FR, PT: Salud mental / Mental health
        r"\bTRANSTORNO",  # PT: Transtornos
        r"\bTRASTORNO",  # ES: Trastornos
        r"\bDISORDER",  # EN: Disorders
        r"\bTROUBLE MENTAL",  # FR: Troubles mentaux
        r"\bPSYCHISCH",  # DE: Mental/Psíquico
        r"\bNERVENDRUCK",  # DE: Sistema nervioso
        # 4. ENFERMEDADES METAXÉNICAS Y CONTROL / VIGILANCIA ENTOMOLÓGICA
        # ES / EN / FR / PT / DE
        r"\bMETAXENIC",  # ES: Metaxénicas
        r"\bVECTOR BORN",  # EN: Vector-borne
        r"\bMALADIE A VECTEUR",  # FR: Maladies à vecteur
        r"\bDOENCAS TRANSMITIDAS POR VETORES\b",  # PT
        r"\bENTOMO",  # Transversal: Entomología / Entomological / Entomologique / Entomologia / Entomologie
        r"\bVECTOR CONTROL\b",  # EN: Control de vectores
        r"\bCONTROLE DE VETORES\b",  # PT
        r"\bINSEKTENKONTROLLE\b",
        r"\bINVESTIGACION CLINICA\b"
        # 5. NEOPLASIAS MALIGNAS Y CÁNCER
        # ES / EN / FR / PT / DE
        r"\bNEOPLAS",  # Transversal: Neoplasias / Neoplasms / Néoplasies / Neoplasien
        r"\bCANCER",  # ES, EN, FR: Cáncer / Cancer
        r"\bCANCEROLOG",  # Cancerología
        r"\bONCOL",  # Transversal: Oncología / Oncology / Oncologie / Oncologia / Onkologie
        r"\bKREBS",  # DE: Cáncer (Krebsforschung, Krebserkrankung)
        # 6. ENFERMEDADES NO TRANSMISIBLES, OBESIDAD Y METABOLISMO
        # ES / EN / FR / PT / DE
        r"\bNON COMMUNICABLE",  # EN: Non-communicable diseases (NCDs)
        r"\bNON TRANSMISSIBLE",  # FR: Non transmissibles
        r"\bNAO TRANSMISSIVE",  # PT: Não transmissíveis
        r"\bNICHTUBERTRAGBAR",  # DE: Nichtübertragbare Krankheiten
        r"\bOBESID",  # ES, PT: Obesidad / Obesidade
        r"\bOBESIT",  # EN, FR: Obesity / Obésité
        r"\bADIPOSI",  # DE: Adipositas (Obesidad en alemán)
        r"\bMETABOL",  # Transversal: Metabolismo / Metabolism / Métabolisme / Metabolismo / Stoffwechsel
        r"\bSTOFFWECHSEL",  # DE: Metabolismo
        # 7. GESTIÓN DE SERVICIOS DE SALUD PÚBLICA
        # ES / EN / FR / PT / DE
        r"\bSALUD PUBLICA\b",  # ES
        r"\bPUBLIC HEALTH\b",  # EN
        r"\bSANTE PUBLIQUE\b",  # FR
        r"\bSAUDE PUBLICA\b",  # PT
        r"\bOFFENTLICHE GESUNDHEIT\b",  # DE: Salud pública
        r"\bGESTION DE SERVICIOS DE SALUD\b",  # ES
        r"\bHEALTH SERVICES MANAGEMENT\b",  # EN
        r"\bGESTION DES SERVICES DE SANTE\b",  # FR
        r"\bGESTÃO DE SERVIÇOS DE SAÚDE\b",  # PT
        r"\bGESUNDHEITSMANAGEMENT\b",  # DE: Gestión de servicios de salud
        ]

}


# 3. Función de categorización optimizada y jerárquica
def categorizar_carrera(carrera):
    carrera_norm = normalizar_texto(carrera)
    if not carrera_norm:
        return "Otros"

    # Evalúa cada área y sus patrones en orden estricto
    for area, patrones in REGLAS_MATCHING.items():
        for patron in patrones:
            if re.search(patron, carrera_norm):
                return area

    return "Otros"


# Aplicación a la columna de tu DataFrame
pronabec['AREA_PRIORIZADA'] = pronabec['CARRERA_PRONABEC'].apply(categorizar_carrera)

pronabec["AREA_PRIORIZADA_BIN"] = pronabec["AREA_PRIORIZADA"].apply(
    lambda x: "NO" if x == "Otros" else "SI"
)


# Se analiza un caso para los registros que contienen biotecnología
caso = pronabec[pronabec["AREA_PRIORIZADA"]=="Biotecnología"]

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

# 5. Estética y escala optimizada (Eje Y amplificado de 0 a 15%)
ax.set_ylim(0, 15)
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

# 2. Definir paleta de colores idéntica a tus gráficos previos
colores_areas = {
    "Ambiente y Cambio Climático": "#D32F2F",  # Rojo
    "CTIsalud": "#388E3C",  # Verde
    "Biotecnología": "#1976D2",  # Azul
    "Inteligencia Artificial": "#7B1FA2",  # Morado
    "Nanotecnología": "#F57C00",  # Naranja
}

# 3. Calcular tabla cruzada en porcentaje por año (Normalizado por fila)
df_micro_cross = (
    pd.crosstab(
        becario_maestria["AÑO_CONVOCATORIA"],
        becario_maestria["AREA_PRIORIZADA"],
        normalize="index",
    )
    * 100
)

# Reordenar columnas para mantener coherencia visual (de mayor a menor relevancia general)
areas_ordenadas = [
    a for a in colores_areas.keys() if a in df_micro_cross.columns
]
df_micro_cross = df_micro_cross[areas_ordenadas]

# 4. Construcción del gráfico de barras apiladas al 100%
fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

x_years = df_micro_cross.index.astype(str)
bottom_val = np.zeros(len(df_micro_cross))

for area_name in areas_ordenadas:
    values = df_micro_cross[area_name].values
    color = colores_areas[area_name]

    bars = ax.bar(
        x_years,
        values,
        bottom=bottom_val,
        label=area_name,
        color=color,
        width=0.65,
        edgecolor="white",
        linewidth=0.8,
    )

    # Añadir porcentaje dentro de los segmentos de barra (solo si la barra es >= 6%)
    for i, val in enumerate(values):
        if val >= 6.0:
            y_pos = bottom_val[i] + val / 2.0
            ax.annotate(
                f"{val:.0f}%",
                (i, y_pos),
                ha="center",
                va="center",
                color="white",
                fontsize=9,
                fontweight="bold",
            )

    bottom_val += values

# 5. Formato y estética
ax.set_ylim(0, 100)
ax.set_ylabel(
    "Distribución Relativa en Áreas Prioritarias (%)",
    fontsize=11,
    fontweight="bold",
    labelpad=10,
)
ax.set_xlabel("Año de Convocatoria", fontsize=11, fontweight="bold", labelpad=10)

ax.set_title(
    "Evolución de la Estructura Interna del Subconjunto Priorizado (2013-2025)",
    fontsize=13,
    fontweight="bold",
    pad=15,
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

# 3. Calcular distribución porcentual por Rango de Edad
df_grouped = becario_maestria.groupby(
    ["RANGO_EDAD", "AREA_PRIORIZADA"], observed=False
).size().unstack(fill_value=0)

# Normalizar por fila (cada rango de edad suma 100%)
df_pct = df_grouped.div(df_grouped.sum(axis=1), axis=0) * 100

# Transformar a formato largo
df_plot = df_pct.reset_index().melt(
    id_vars="RANGO_EDAD", var_name="AREA_PRIORIZADA", value_name="Porcentaje"
)

# 4. Configuración de estilo gráfico
sns.set_style("whitegrid")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.figure(figsize=(12, 6), dpi=300)

# 5. Creación del gráfico con la paleta homologada
ax = sns.barplot(
    data=df_plot,
    x="RANGO_EDAD",
    y="Porcentaje",
    hue="AREA_PRIORIZADA",
    palette=colores_areas,  # <--- Paleta homologada
    hue_order=[
        "Ambiente y Cambio Climático",
        "CTIsalud",
        "Biotecnología",
        "Inteligencia Artificial",
        "Nanotecnología",
    ],
)

# 6. Etiquetas de valores sobre cada barra
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(
            f"{height:.0f}%",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            xytext=(0, 3),
            textcoords="offset points",
            fontsize=12,
            fontweight="bold",
        )

# 7. Formato de títulos y ejes
ax.set_xlabel("Rango de Edades", fontsize=12, fontweight="bold", labelpad=10)
ax.set_ylabel("Porcentaje (%)", fontsize=12, fontweight="bold", labelpad=10)

# Ajustar límite Y para dar margen a las etiquetas sobre las barras
plt.ylim(0, max(df_plot["Porcentaje"]) * 1.15)

# Leyenda configurada
plt.legend(
    title="Área Priorizada",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    frameon=True,
    facecolor="#FFFFFF",
    edgecolor="#CCCCCC",
    fontsize=10,
    title_fontsize=11,
)

sns.despine(top=True, right=True)
plt.tight_layout()
plt.show()

###############################################################################
# Se calcula la composición de programas de maestria asociados con áreas
# priorizadas por Género
###############################################################################
# 1. Definir la paleta exacta de tu segundo gráfico
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

# 3. Calcular porcentajes dentro de cada Género
df_genero = (
    pd.crosstab(
        becario_micro["SEXO"],
        becario_micro["AREA_PRIORIZADA"],
        normalize="index",
    )
    * 100
).reset_index()

# Transformar a formato largo para Seaborn
df_genero_melted = pd.melt(
    df_genero,
    id_vars=["SEXO"],
    var_name="AREA_PRIORIZADA",
    value_name="Porcentaje",
)

# 4. Estilo de figura
sns.set_style("whitegrid")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.figure(figsize=(10, 6), dpi=300)

# 5. Gráfico de barras agrupadas usando la paleta personalizada
ax = sns.barplot(
    data=df_genero_melted,
    x="SEXO",
    y="Porcentaje",
    hue="AREA_PRIORIZADA",
    palette=colores_areas,  # <--- Paleta homologada
    hue_order=[
        "Ambiente y Cambio Climático",
        "CTIsalud",
        "Biotecnología",
        "Inteligencia Artificial",
        "Nanotecnología",
    ],
)

# 6. Agregar etiquetas porcentuales sobre las barras
for p in ax.patches:
    height = p.get_height()
    if height > 0:  # Mostrar solo si la barra tiene valor
        ax.annotate(
            f"{height:.0f}%",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            xytext=(0, 3),
            textcoords="offset points",
            fontsize=12,
            fontweight="bold",
        )

# 7. Formato y límites
plt.ylim(0, max(df_genero_melted["Porcentaje"]) * 1.15)
plt.xlabel("Género", fontsize=12, fontweight="bold", labelpad=10)
plt.ylabel("Porcentaje (%)", fontsize=12, fontweight="bold", labelpad=10)

plt.legend(
    title="Área priorizada",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
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

# Se cuenta la combinación considerando los tres atributos
df_pivot = (
    becario_maestria.groupby(
        ["AREA_PRIORIZADA", "RANGO_EDAD", "SEXO"], observed=False
    )
    .size()
    .unstack(level=["RANGO_EDAD", "SEXO"], fill_value=0)
)

# Normalizar por columna para obtener el % intragrupo (cada columna suma 100%)
heatmap_data = df_pivot.div(df_pivot.sum(axis=0), axis=1) * 100

# Reordenar las columnas de forma lógica (Jóvenes -> Adultos | Femenino -> Masculino)
column_order = [
    ("20 - 29 años", "FEMENINO"),
    ("20 - 29 años", "MASCULINO"),
    ("30 - 39 años", "FEMENINO"),
    ("30 - 39 años", "MASCULINO"),
    ("40 - 50 años", "FEMENINO"),
    ("40 - 50 años", "MASCULINO"),
]

# Filtrar columnas existentes por si algún grupo no tiene datos
column_order = [col for col in column_order if col in heatmap_data.columns]
heatmap_data = heatmap_data[column_order]

# Renombrar etiquetas de columnas para una presentación impecable
formatted_columns = [f"{edad}\n({sexo[:3].upper()})" for edad, sexo in column_order]

# 2. CONFIGURACIÓN VISUAL DEL HEATMAP
plt.figure(figsize=(10, 6), dpi=300)
sns.set_style("white")

# Paleta discreta de tonos verdes/azules para lectura rápida
ax = sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".1f",
    cmap="YlGnBu",
    linewidths=0.8,
    linecolor="white",
    cbar_kws={"label": "Proporción Intragrupo (%)", "shrink": 0.8},
    annot_kws={"size": 14, "weight": "bold"},
)

# Ajustar etiquetas
ax.set_xticklabels(formatted_columns, rotation=0, ha="center", fontsize=9.5)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9.5)

#plt.title(
    #"Matriz de Distribución Disciplinar Intragrupo por Edad y Género (%)",
    #fontsize=12,
    #fontweight="bold",
    #pad=20,
#)
plt.xlabel("Cohorte Demográfica (Rango de Edad / Género)", fontsize=10, labelpad=12, fontweight="bold")
plt.ylabel("Área priorizada", fontsize=10, labelpad=12, fontweight="bold")

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
        becario_doctorado[col_anio], becario_doctorado[col_target], normalize="index"
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

# 5. Estética y escala optimizada (Eje Y amplificado de 0 a 15%)
ax.set_ylim(0, 15)
ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels, fontsize=9.5)
ax.set_ylabel("Participación (%)", fontsize=10, labelpad=10, fontweight="bold")
ax.set_xlabel("Año de Convocatoria", fontsize=10, labelpad=10, fontweight="bold")

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

becario_doctorado["RANGO_EDAD"] = pd.cut(
    becario_doctorado["EDADBASES"], bins=bins, labels=labels, right=True
)

# 2. CALCULO MACRO (ESTRUCTURAL / PARTE-TODO)
# Universo total de becarios de maestría (Denominador N)
N_total = len(becario_doctorado)

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
becario_maestria.SEXO.value_counts(normalize=True).round(2)*100

###############################################################################
# Se calcula un gráfico apilado para mostrar la participación de cada area priorizada
# entre ellas
###############################################################################

# 2. Definir paleta de colores idéntica a tus gráficos previos
colores_areas = {
    "Ambiente y Cambio Climático": "#D32F2F",  # Rojo
    "CTIsalud": "#388E3C",  # Verde
    "Biotecnología": "#1976D2",  # Azul
    "Inteligencia Artificial": "#7B1FA2",  # Morado
    "Nanotecnología": "#F57C00",  # Naranja
}

# 3. Calcular tabla cruzada en porcentaje por año (Normalizado por fila)
df_micro_cross = (
    pd.crosstab(
        becario_doctorado["AÑO_CONVOCATORIA"],
        becario_doctorado["AREA_PRIORIZADA"],
        normalize="index",
    )
    * 100
)

# Reordenar columnas para mantener coherencia visual (de mayor a menor relevancia general)
areas_ordenadas = [
    a for a in colores_areas.keys() if a in df_micro_cross.columns
]
df_micro_cross = df_micro_cross[areas_ordenadas]

# 4. Construcción del gráfico de barras apiladas al 100%
fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

x_years = df_micro_cross.index.astype(str)
bottom_val = np.zeros(len(df_micro_cross))

for area_name in areas_ordenadas:
    values = df_micro_cross[area_name].values
    color = colores_areas[area_name]

    bars = ax.bar(
        x_years,
        values,
        bottom=bottom_val,
        label=area_name,
        color=color,
        width=0.65,
        edgecolor="white",
        linewidth=0.8,
    )

    # Añadir porcentaje dentro de los segmentos de barra (solo si la barra es >= 6%)
    for i, val in enumerate(values):
        if val >= 6.0:
            y_pos = bottom_val[i] + val / 2.0
            ax.annotate(
                f"{val:.0f}%",
                (i, y_pos),
                ha="center",
                va="center",
                color="white",
                fontsize=9,
                fontweight="bold",
            )

    bottom_val += values

# 5. Formato y estética
ax.set_ylim(0, 100)
ax.set_ylabel(
    "Distribución Relativa en Áreas Prioritarias (%)",
    fontsize=11,
    fontweight="bold",
    labelpad=10,
)
ax.set_xlabel("Año de Convocatoria", fontsize=11, fontweight="bold", labelpad=10)

ax.set_title(
    "Evolución de la Estructura Interna del Subconjunto Priorizado (2013-2025)",
    fontsize=13,
    fontweight="bold",
    pad=15,
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

# 3. Calcular distribución porcentual por Rango de Edad
df_grouped = becario_doctorado.groupby(
    ["RANGO_EDAD", "AREA_PRIORIZADA"], observed=False
).size().unstack(fill_value=0)

# Normalizar por fila (cada rango de edad suma 100%)
df_pct = df_grouped.div(df_grouped.sum(axis=1), axis=0) * 100

# Transformar a formato largo
df_plot = df_pct.reset_index().melt(
    id_vars="RANGO_EDAD", var_name="AREA_PRIORIZADA", value_name="Porcentaje"
)

# 4. Configuración de estilo gráfico
sns.set_style("whitegrid")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.figure(figsize=(12, 6), dpi=300)

# 5. Creación del gráfico con la paleta homologada
ax = sns.barplot(
    data=df_plot,
    x="RANGO_EDAD",
    y="Porcentaje",
    hue="AREA_PRIORIZADA",
    palette=colores_areas,  # <--- Paleta homologada
    hue_order=[
        "Ambiente y Cambio Climático",
        "CTIsalud",
        "Biotecnología",
        "Inteligencia Artificial",
        "Nanotecnología",
    ],
)

# 6. Etiquetas de valores sobre cada barra
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(
            f"{height:.0f}%",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            xytext=(0, 3),
            textcoords="offset points",
            fontsize=12,
            fontweight="bold",
        )

# 7. Formato de títulos y ejes
ax.set_xlabel("Rango de Edades", fontsize=12, fontweight="bold", labelpad=10)
ax.set_ylabel("Porcentaje (%)", fontsize=12, fontweight="bold", labelpad=10)

# Ajustar límite Y para dar margen a las etiquetas sobre las barras
plt.ylim(0, max(df_plot["Porcentaje"]) * 1.15)

# Leyenda configurada
plt.legend(
    title="Área Priorizada",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    frameon=True,
    facecolor="#FFFFFF",
    edgecolor="#CCCCCC",
    fontsize=10,
    title_fontsize=11,
)

sns.despine(top=True, right=True)
plt.tight_layout()
plt.show()

###############################################################################
# Se calcula la composición de programas de doctorado asociados con áreas
# priorizadas por Género
###############################################################################
# 1. Definir la paleta exacta de tu segundo gráfico
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

# 3. Calcular porcentajes dentro de cada Género
df_genero = (
    pd.crosstab(
        becario_micro["SEXO"],
        becario_micro["AREA_PRIORIZADA"],
        normalize="index",
    )
    * 100
).reset_index()

# Transformar a formato largo para Seaborn
df_genero_melted = pd.melt(
    df_genero,
    id_vars=["SEXO"],
    var_name="AREA_PRIORIZADA",
    value_name="Porcentaje",
)

# 4. Estilo de figura
sns.set_style("whitegrid")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.figure(figsize=(10, 6), dpi=300)

# 5. Gráfico de barras agrupadas usando la paleta personalizada
ax = sns.barplot(
    data=df_genero_melted,
    x="SEXO",
    y="Porcentaje",
    hue="AREA_PRIORIZADA",
    palette=colores_areas,  # <--- Paleta homologada
    hue_order=[
        "Ambiente y Cambio Climático",
        "CTIsalud",
        "Biotecnología",
        "Inteligencia Artificial",
        "Nanotecnología",
    ],
)

# 6. Agregar etiquetas porcentuales sobre las barras
for p in ax.patches:
    height = p.get_height()
    if height > 0:  # Mostrar solo si la barra tiene valor
        ax.annotate(
            f"{height:.0f}%",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            xytext=(0, 3),
            textcoords="offset points",
            fontsize=12,
            fontweight="bold",
        )

# 7. Formato y límites
plt.ylim(0, max(df_genero_melted["Porcentaje"]) * 1.15)
plt.xlabel("Género", fontsize=12, fontweight="bold", labelpad=10)
plt.ylabel("Porcentaje (%)", fontsize=12, fontweight="bold", labelpad=10)

plt.legend(
    title="Área priorizada",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
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

# Se cuenta la combinación considerando los tres atributos
df_pivot = (
    becario_doctorado.groupby(
        ["AREA_PRIORIZADA", "RANGO_EDAD", "SEXO"], observed=False
    )
    .size()
    .unstack(level=["RANGO_EDAD", "SEXO"], fill_value=0)
)

# Normalizar por columna para obtener el % intragrupo (cada columna suma 100%)
heatmap_data = df_pivot.div(df_pivot.sum(axis=0), axis=1) * 100

# Reordenar las columnas de forma lógica (Jóvenes -> Adultos | Femenino -> Masculino)
column_order = [
    ("20 - 29 años", "FEMENINO"),
    ("20 - 29 años", "MASCULINO"),
    ("30 - 39 años", "FEMENINO"),
    ("30 - 39 años", "MASCULINO"),
    ("40 - 50 años", "FEMENINO"),
    ("40 - 50 años", "MASCULINO"),
]

# Filtrar columnas existentes por si algún grupo no tiene datos
column_order = [col for col in column_order if col in heatmap_data.columns]
heatmap_data = heatmap_data[column_order]

# Renombrar etiquetas de columnas para una presentación impecable
formatted_columns = [f"{edad}\n({sexo[:3].upper()})" for edad, sexo in column_order]

# 2. CONFIGURACIÓN VISUAL DEL HEATMAP
plt.figure(figsize=(10, 6), dpi=300)
sns.set_style("white")

# Paleta discreta de tonos verdes/azules para lectura rápida
ax = sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".1f",
    cmap="YlGnBu",
    linewidths=0.8,
    linecolor="white",
    cbar_kws={"label": "Proporción Intragrupo (%)", "shrink": 0.8},
    annot_kws={"size": 14, "weight": "bold"},
)

# Ajustar etiquetas
ax.set_xticklabels(formatted_columns, rotation=0, ha="center", fontsize=9.5)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9.5)

#plt.title(
    #"Matriz de Distribución Disciplinar Intragrupo por Edad y Género (%)",
    #fontsize=12,
    #fontweight="bold",
    #pad=20,
#)
plt.xlabel("Cohorte Demográfica (Rango de Edad / Género)", fontsize=10, labelpad=12, fontweight="bold")
plt.ylabel("Área priorizada", fontsize=10, labelpad=12, fontweight="bold")

plt.tight_layout()
plt.show()


