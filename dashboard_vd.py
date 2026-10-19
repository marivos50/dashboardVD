# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 13:31:48 2026

@author: Mari
"""

# -*- coding: utf-8 -*-
"""
Dashboard de Visualización de Datos - Práctica Final
Creado: 2026
@author: Mari
Versión 3.0 - Integración completa con resultados del notebook
"""

import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sklearn
import seaborn as sns
import networkx as nx
import os
import json

# ============================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================
st.set_page_config(
    page_title="Dashboard VD - Práctica Final",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado mejorado
st.markdown("""
<style>
    /* Fondo general */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Estilo para las cajas de contenido */
    .dashboard-box {
        background-color: white;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Títulos de sección */
    .section-title {
        color: #1a7a3a;
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 20px;
        padding-bottom: 10px;
        border-bottom: 2px solid #1a7a3a;
    }
    
    .subsection-title {
        color: #2c3e50;
        font-size: 16px;
        font-weight: bold;
        margin-top: 15px;
        margin-bottom: 10px;
    }
    
    /* Texto de features mejorado */
    .feature-text {
        color: #2c3e50;
        font-size: 13px;
        line-height: 1.8;
    }
    
    .feature-text li {
        margin-bottom: 5px;
    }
    
    .feature-text .highlight {
        color: #1a7a3a;
        font-weight: bold;
    }
    
    /* Lista sin viñetas */
    .no-bullets {
        list-style-type: none;
        padding-left: 0;
    }
    
    /* Métricas interactivas */
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #dee2e6;
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #1a7a3a;
        margin: 5px 0;
    }
    
    .metric-label {
        font-size: 12px;
        color: #6c757d;
    }
    
    /* Estilo para tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: white;
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #dee2e6;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        font-size: 14px;
        font-weight: 500;
        border-radius: 8px;
        padding: 0 20px;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #1a7a3a;
        color: white;
    }
    
    /* Tarjetas de información */
    .info-card {
        background-color: #e7f5ff;
        border-left: 4px solid #0d6efd;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    .success-card {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    .warning-card {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    /* Botones personalizados */
    .stButton > button {
        background-color: #1a7a3a;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stButton > button:hover {
        background-color: #155d2e;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# FUNCIONES DE CARGA DE DATOS
# ============================================
@st.cache_data
def load_dataset():
    """Carga el dataset principal"""
    filepath = "./DS_VISU_MICD.xlsx"
    try:
        df = pd.read_excel(filepath)
        return df
    except FileNotFoundError:
        st.error(f"No se encontró el archivo: {filepath}")
        # Intentar ruta relativa
        df = pd.read_excel("./DS_VISU_MICD.xlsx")
        return df

@st.cache_data
def load_clean_dataset():
    """Carga el dataset limpio (df_visu) desde resultados"""
    results_dir = Path("./datos_vd/resultados")
    visu_path = results_dir / "df_visu.pkl"
    
    if visu_path.exists():
        with open(visu_path, 'rb') as f:
            df_visu = pickle.load(f)
        return df_visu
    else:
        st.warning("⚠️ No se encontró el dataset limpio (df_visu.pkl)")
        return None
    
@st.cache_data
def load_ordenado_dataset():
    """Carga el dataset df_ordenado desde resultados"""
    results_dir = Path("./datos_vd/resultados")
    ord_path = results_dir / "df_ordenado.pkl"
    figures_dir=Path("./datos_vd/figuras")
    
    if ord_path.exists():
        with open(ord_path, 'rb') as f:
            df_ordenado = pickle.load(f)
        return df_ordenado
    else:
        st.warning("⚠️ No se encontró el dataset df_ordenado)")
        return None

@st.cache_data
def load_analysis_results():
    """Carga todos los resultados del análisis"""
    results = {}
    
    # Directorio donde están los resultados
    results_dir = Path("./datos_vd/resultados")
    figures_dir = Path("./datos_vd/figuras")
    
    # Cargar punto1_basica.pkl 
    pkl_path = results_dir / "punto1_basica.pkl"
    if pkl_path.exists():
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            # data es un diccionario con TODAS esas claves
            results.update(data)  # ← Esto carga TODO el diccionario
    else:
        st.warning("No se encontró punto1_basica.pkl")
    
    # Cargar punto1_detallada.pkl
    pkl_path = results_dir / "punto1_detallada.pkl"
    if pkl_path.exists():
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            results.update(data)  # Carga TODO el diccionario
    else:
        st.error("No se encontró punto1_basica.pkl")
        
    # Cargar balanceo_edad_sexo.pkl
    pkl_path = results_dir / "punto2.pkl"
    if pkl_path.exists():
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
    else:
        st.error("No se encontró punto2.pkl")        
    
    # Cargar diagnostico
    pkl_path = results_dir / "diagnostico.pkl"
    if pkl_path.exists():
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            results.update(data)  # Carga TODO el diccionario
    else:
        st.warning("No se encontró diagnostico.pkl")
    
    pkl_path = results_dir / "outliers_features.pkl"
    if pkl_path.exists():
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            # data es un diccionario con TODAS esas claves
            results.update(data)  # ← Esto carga TODO el diccionario
    else:
        st.warning("No se encontró outliers_features.pkl")
    
    # Cargar arbol
    pkl_path = results_dir / "punto3_arbol.pkl"
    if pkl_path.exists():
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            results.update(data)  # Carga TODO el diccionario
    else:
        st.error("No se encontró punto3_arbol.pkl")
    
    # Cargar consistencia.pkl
    #pkl_path = results_dir / "consistencia.pkl"
    #if pkl_path.exists():
    #    with open(pkl_path, 'rb') as f:
    #        data = pickle.load(f)
    #        results.update(data)  # Carga TODO el diccionario
    #else:
    #    st.error("No se encontró consistencia.pkl")


    
    #Cargar punto4_scatter.pkl
    pkl_path = results_dir / "punto4_scatter.pkl"
    if pkl_path.exists():
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            results.update(data)  # Carga TODO el diccionario
    else:
        st.error("No se encontró punto4_scatter.pkl") 
        
    #Cargar punto5_casos_dificiles.pkl
    pkl_path = results_dir / "punto5_casos_dificiles.pkl"
    if pkl_path.exists():
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            results.update(data)  # Carga TODO el diccionario
    else:
        st.error("No se encontró punto5_casos_dificiles.pkl")
        
    
    results['figures'] = {}
    if figures_dir.exists():
        # Buscar imágenes PNG
        for fig_path in figures_dir.glob("*.png"):
            results['figures'][fig_path.stem] = str(fig_path)
        
        # Buscar gráficos HTML (Plotly)
        for fig_path in figures_dir.glob("*.html"):
            results['figures'][fig_path.stem] = str(fig_path)
        
        # Buscar imágenes JPG/JPEG
        for fig_path in figures_dir.glob("*.jpg"):
            results['figures'][fig_path.stem] = str(fig_path)
        for fig_path in figures_dir.glob("*.jpeg"):
            results['figures'][fig_path.stem] = str(fig_path)

    return results

@st.cache_data
def load_predictions():
    """Carga las predicciones del modelo"""
    pred_path = Path("./resultados/predicciones_modelo.pkl")
    if pred_path.exists():
        with open(pred_path, 'rb') as f:
            return pickle.load(f)
    return None



# ============================================
# Carga de Datos: dataset y resultadosy figuras
# ============================================
df_raw = load_dataset()

df_visu = load_clean_dataset()
df_ordenado=load_ordenado_dataset()
results = load_analysis_results()

outliers_df_summary = results['df_outliers_summary']
outliers_df = results['df_outliers_tabla']
features_outliers = results['features_outliers']

# Si df_visu está cargado, añadirlo a results para usarlo en el dashboard
if df_visu is not None:
    results['df_visu'] = df_visu
predicciones = load_predictions()

# Definir paletas de colores
colores_alternas ={0: '#F5F9FC ' ,1: '#FAFCFF '}  #azul hielo, blanco azulado

colors_si_no={0 : '#1D8224', 1 : '#F40B1B'} # color verde-rojo

colors_sexo = {1: '#7FE817', 2: '#FFFF00'}

colors_dx = { 1 : '#FFFF00',  2 : '#003BFF',  3 : '#8A6E19',  4 : '#FF00FF', 5 : '#7FE817'}
			#amarillo,azul, café, violetta,verde

colors_prex= {1 : '#D1D100',  2 : '#0031D1',  3 : '#703E1B',  4 : '#D100D1', 5 : '#69Be13'}
    #mismas como de dx, pero mas oscuras
    
colors_diag={'ED1': '#F7F72A','ED2' : '#90F72A', 'ED3':'#2AF7F7'}
# amarillo,verde,azul clarito

# colores según nivel de comun-muy raro
colors_grupos = {
    'comun': '#2196F3',        # azul claro (frecuente)
    'poco comun': '#1976D2',   # azul medio
    'raro': '#0D47A1',         # azul oscuro
    'muy raro': '#880E4F'      # púrpura oscuro (raro = especial)
}

# Definir paleta de rareza (mismos colores que en el árbol)
colors_rareza = {
    'comun': '#4CAF50',        # Verde
    'poco comun': '#8BC34A',   # Verde claro / amarillento
    'raro': '#FF9800',         # Naranja
    'muy raro': '#F44336'      # Rojo
}

# Colores
colors_niveles = {
    'Facil': '#008000',
    'Moderado': '#9ACD32',
    'Difícil': '#FFA500',
    'Muy Difícil': '#FF0000',
    'Extremadamente Difícil': '#8B0000'
}

pd.set_option('display.float_format', '{:.2f}'.format)


# ============================================
# TABS EN ORDEN LÓGICO
# ============================================
tab1, tab2, tab3, tab4, tab5,tab6  = st.tabs([
    "🔍 Vista Raw (Problemas)", 
    "🧹 Limpieza & Preprocesamiento", 
    "📊 Exploración Edad y Sexo",
    "Diagnósticos",
    "Predicciones",
    "Casos Difíciles"    
])




# TAB 1: VISTA RAW
# Fila 1
with tab1:
       
    st.header("🔍 Diagnóstico de Calidad de Datos")
    st.caption("Identificación de problemas en datos originales")

    
    
    # Fila 1
    
    with st.container():
        # Título que ocupa todo el ancho
        st.subheader("📋 Muestra de Datos")
        
        # Crear 4 columnas pero la tabla ocupará las 3 derechas
        col_left,col_media, col_right = st.columns([1,1,2])
        
        with col_left:
            st.write("**Información del Dataset:**")
            st.write(f"- 🔢 {len(df_raw)} filas")
            st.write(f"- 📊 {len(df_raw.columns)} columnas")
            st.write(f"- 📈 {df_raw.select_dtypes(include=['number']).shape[1]} columnas numéricas")
            st.write(f"- 📝 {df_raw.select_dtypes(include=['object']).shape[1]} columnas categóricas")
            st.write(f"- ❌ **Valores Nulos:** {results['total_faltantes']}")
            st.write(f"- 🔄 **Registros Duplicados:** {results['duplicados_filas']}")
            st.write(f"- 🆔 **IDs Duplicados:** {results['duplicados_id']}")
            

        with col_media:
            st.markdown("**Tipos de datos:**")
            tipo_df = df_raw.dtypes.reset_index()
            tipo_df.columns = ['Feature', 'Tipo']
            
            # Crear columna auxiliar para el índice
            tipo_df['_index'] = tipo_df.index
            
            styled_df = tipo_df.style.apply(lambda x: 
                ['background-color: ' + colores_alternas[0] if i % 2 == 0 
                 else 'background-color: ' + colores_alternas[1] 
                 for i in range(len(x))], axis=0)
            
            st.dataframe(styled_df, use_container_width=True)
                                
                
        
        with col_right:
            # Tabla completa sin dividir
            st.markdown("**Dataset df_raw:**")
           
            # Crear columna auxiliar para el índice
            df_raw['_index'] = df_raw.index
            
            styled_df = df_raw.style.apply(lambda x: 
                ['background-color: ' + colores_alternas[0] if i % 2 == 0 
                 else 'background-color: ' + colores_alternas[1] 
                 for i in range(len(x))], axis=0)
            
            st.dataframe(styled_df, use_container_width=True)
    
    st.divider()
    
    # Fila2 Outliers
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container():
            st.subheader("Outliers")
    
        
            if 'outliers' in results and results['outliers']:
                outliers_data = results['outliers']
                
                # Si es un diccionario directamente
                parsed_data = []
                for feature, info in outliers_data.items():
                    if isinstance(info, dict):
                        parsed_data.append({
                            'Variable': feature,
                            'Outliers': info.get('outliers', 0),
                            'Porcentaje': f"{info.get('pct', 0):.2f}%",
                            'Límite Inferior': f"{info.get('lower_bound', 0):.2f}",
                            'Límite Superior': f"{info.get('upper_bound', 0):.2f}"
                        })
                
                df_outliers = pd.DataFrame(parsed_data)
                #st.dataframe(df_outliers, use_container_width=True, hide_index=True)
                # Crear columna auxiliar para el índice
                df_outliers['_index'] = df_outliers.index
                
                styled_df = df_outliers.style.apply(lambda x: 
                    ['background-color: ' + colores_alternas[0] if i % 2 == 0 
                     else 'background-color: ' + colores_alternas[1] 
                     for i in range(len(x))], axis=0)
                
                st.dataframe(styled_df, use_container_width=True)
                
                
    with col2:
        
        with st.container():
            st.markdown('<h3 class="subsection-title">📊 Visualización de Outliers</h3>', unsafe_allow_html=True)
            
            if 'outliers' in results and results['outliers'] is not None:
                with st.container():
                    outliers = results['outliers']
                    
                    # Obtener lista de features
                    features_disponibles = list(outliers.keys())
                    
                    # Verificar qué features tienen imágenes guardadas
                    figures_dir=Path("./datos_vd/figuras")
                    features_con_imagen = [f for f in features_disponibles if (figures_dir / f'{f}_box.png').exists()]
                    
                    # Selector de feature
                    feature_seleccionado = st.selectbox(
                        "Selecciona un feature:",
                        options=features_con_imagen if features_con_imagen else features_disponibles,
                        key="feature_boxplot"
                    )
                
                if feature_seleccionado:
                    # boxplot
                    with st.container():
                        image_path = figures_dir / f'{feature_seleccionado}_box.png'
                        
                        
                        st.image(
                                str(image_path), 
                                caption=f'Boxplot de {feature_seleccionado}', 
                                use_container_width=True
                                )
                        

    
    # Crear 6 columnas iguales para todos los valores
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    #with st.container():
    stats = outliers[feature_seleccionado]
    with col1:
        color_outliers = colors_si_no[1] if stats['outliers'] == 0 else colors_si_no[0]
        st.markdown(f"""
        <div style="text-align: center; background: #f8f9fa; padding: 10px; border-radius: 8px; border: 1px solid #dee2e6; height: 100%;">
            <div style="font-size: 11px; color: #6c757d;">⚠️ Outliers in {feature_seleccionado}</div>
            <div style="font-size: 22px; font-weight: bold; color: {color_outliers};">
                {stats['outliers']}
            </div>
            <div style="font-size: 11px; color: #6c757d;">({stats['pct']:.2f}%)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="text-align: center; background: #f8f9fa; padding: 10px; border-radius: 8px; border: 1px solid #dee2e6; height: 100%;">
            <div style="font-size: 11px; color: #6c757d;">📊 Q1</div>
            <div style="font-size: 20px; font-weight: bold; color: #1a7a3a;">{stats['Q1']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="text-align: center; background: #f8f9fa; padding: 10px; border-radius: 8px; border: 1px solid #dee2e6; height: 100%;">
            <div style="font-size: 11px; color: #6c757d;">📈 Q3</div>
            <div style="font-size: 20px; font-weight: bold; color: #1a7a3a;">{stats['Q3']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div style="text-align: center; background: #f8f9fa; padding: 10px; border-radius: 8px; border: 1px solid #dee2e6; height: 100%;">
            <div style="font-size: 11px; color: #6c757d;">📊 IQR</div>
            <div style="font-size: 20px; font-weight: bold; color: #6f42c1;">{stats['IQR']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        
        st.markdown(f"""
        <div style="text-align: center; background: #e7f5ff; padding: 10px; border-radius: 8px; border: 1px solid #dee2e6; height: 100%;">
            <div style="font-size: 10px; color: #6c757d;">⬇️ Límite Inferior</div>
            <div style="font-size: 18px; font-weight: bold; color: #0d6efd;">{stats['limite_inferior']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col6:
        st.markdown(f"""
        <div style="text-align: center; background: #e7f5ff; padding: 10px; border-radius: 8px; border: 1px solid #dee2e6; height: 100%;">
            <div style="font-size: 10px; color: #6c757d;">⬆️ Límite Superior</div>
            <div style="font-size: 18px; font-weight: bold; color: #0d6efd;">{stats['limite_superior']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
                    
                    
                
# TAB2: Limpieza Y Preprocesamiento                         
# Fila 1
with tab2:
    
    # Verificar si df_visu está cargado
    if df_visu is not None:

        # Resultados de la limpieza, métricas

        st.markdown('<h3 class="subsection-title">📊 Resultados de la Limpieza</h3>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            with st.container(border=True):
                st.metric(
                    "📋 Registros finales",
                    df_visu.shape[0]
                )
        
        with col2:
            with st.container(border=True):
                st.metric(
                    "📊 Features finales",
                    df_visu.shape[1]
                )
        
        with col3:
            with st.container(border=True):
                features_eliminadas = df_raw.shape[1] - df_visu.shape[1]
                st.metric(
                    "🗑️ Features eliminadas",
                    features_eliminadas,
                    delta=f"-{features_eliminadas}" if features_eliminadas > 0 else "0"
                )
        
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                # Valores nulos después de la limpieza
                nulos_finales = df_visu.isnull().sum().sum()
                st.metric(
                    "✅ Valores Nulos finales",
                    nulos_finales,
                    delta="0" if nulos_finales == 0 else f"{nulos_finales}"
                )
        with col2:
            with st.container(border=True):
                #Estandarizados
                st.metric(
                    " Datos númericos (menos ID)",
                    "Normalización",
                     "Z-Score "
                )
        
        st.divider()
        
        
        # Comparativa Antes vs. Después
        # Fila 2
        st.markdown('<h3 class="subsection-title">📋 Comparativa: Datos Originales vs Datos Limpios</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):

                
                st.markdown("**📄 Datos Originales (df_raw)**")
                #st.dataframe(df_raw.head(5), use_container_width=True)
                df=df_raw.head(5)
                # Crear columna auxiliar para el índice
                df['_index'] = df.index
                
                styled_df = df.style.apply(lambda x: 
                    ['background-color: ' + colores_alternas[0] if i % 2 == 0 
                     else 'background-color: ' + colores_alternas[1] 
                     for i in range(len(x))], axis=0)
                
                st.dataframe(styled_df, use_container_width=True)
                st.caption(f"Shape: {df_raw.shape} | {df_raw.shape[1]} columnas")
                
        with col2:
            with st.container(border=True):
                
                st.markdown("**✅ Datos Limpios (df_visu)**")
                
                # Crear columna auxiliar para el índice
                df=df_visu.head(5)
                df['_index'] = df.index
                
                styled_df = df.style.apply(lambda x: 
                    ['background-color: ' + colores_alternas[0] if i % 2 == 0 
                     else 'background-color: ' + colores_alternas[1] 
                     for i in range(len(x))], axis=0)
                
                st.dataframe(styled_df, use_container_width=True)
                st.caption(f"Shape: {df_visu.shape} | {df_visu.shape[1]} columnas")
                
            
        st.divider()
        
        # Columnas originales vs finales
        # Fila 3
        col1 = st.columns(1)[0]
        with col1:
            with st.container(border=True):
                # Detectar columnas eliminadas
                
                # ============================================
                # DETECCIÓN DE COLUMNAS ELIMINADAS (CORREGIDO)
                # ============================================
                
                # Obtener conjuntos de columnas
                columnas_originales = set(df_ordenado.columns)
                columnas_finales = set(df_visu.columns)
                
                # Crear lista de nombres en mayúsculas para comparación
                cols_final_upper = [col.upper() for col in columnas_finales]
                
                # Detectar columnas eliminadas (case-insensitive)
                columnas_eliminadas = []
                for col in columnas_originales:
                    # Si la columna original NO está en las finales (comparando en mayúsculas)
                    if col.upper() not in cols_final_upper:
                        columnas_eliminadas.append(col)  # tiene nombres del df_ordenados
                
                
                
                st.markdown("Features eliminados")                
                # Mostrar columnas originales en grid
                cols_orig = st.columns(10)
                for i, col in enumerate(df_ordenado):
                    with cols_orig[i % 10]:

                        if col.upper() in columnas_eliminadas:
                            st.markdown(f'''
                            <span style="position: relative; 
                                         display: inline-block;
                                         background-color: #f8f9fa; 
                                         color:{colors_si_no[1]};            
                                         padding: 4px 10px; 
                                         border-radius: 12px; 
                                         font-size: 13px; 
                                         font-weight: 500;
                                         border: 1px solid #dee2e6;">
                                {col}
                                
                            </span>
                            ''', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<font color={colors_si_no[0]}>{col}', unsafe_allow_html=True)
                st.caption(f"Total: {len(df_raw.columns)} columnas{len(df_ordenado.columns)}")               

        st.divider()
     
        # Cambios realizadas
        # Fila 4
        st.markdown('<h3 class="subsection-title">📝 Cambios Realizados</h3>', unsafe_allow_html=True)
        
        # Detectar columnas eliminadas
        columnas_originales = set(df_raw.columns)
        columnas_finales = set(df_visu.columns)
        columnas_eliminadas = columnas_originales - columnas_finales
        
        # Detectar columnas renombradas (ej: EDAD -> Edad)
        columnas_renombradas = {}
        for col_orig in columnas_originales:
            if col_orig not in columnas_finales:
                # Buscar si hay una columna similar en finales
                for col_final in columnas_finales:
                    if col_orig.lower() == col_final.lower():
                        columnas_renombradas[col_orig] = col_final
                        break
        
        # Quitar renombradas de eliminadas
        columnas_eliminadas = columnas_eliminadas - set(columnas_renombradas.keys())
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Valores nulos
            nulos_original = df_raw.isnull().sum().sum()
            nulos_final = df_visu.isnull().sum().sum()
            # Solamente una columna renombrado
            for orig, final in columnas_renombradas.items():
                orig=orig
                final=final
            st.markdown(f"""
            <div class="dashboard-box">
                <h4 style="color: #1a7a3a; margin-top: 0;">✅ Acciones Realizadas</h4>
                <ul style="list-style-type: none; padding-left: 0; line-height: 2;">
                    <li>✓ <strong>Registros:</strong> {df_raw.shape[0]} → {df_visu.shape[0]}</li>
                    <li>✓ <strong>No se ha eliminado ningún registro.</strong></li>
                    <li>✓ <strong>Features:</strong> {df_raw.shape[1]} → {df_visu.shape[1]}</li>
                    <li>✓ <strong>Features eliminadas:</strong> {len(columnas_eliminadas)}</li>
                    <li>✓ <strong>Features renombradas:</strong></li>             
                    <li>✓ <strong>EDAD → Edad</strong></li>
                    <li>✓ <strong>Valores nulos:</strong> {nulos_original} → {nulos_final}</li>
                    <li>✓ <strong>No había valores nulos</strong></li>
            """, unsafe_allow_html=True)
            
            
                   
            
                  
        with col2:
            if columnas_eliminadas:
                st.markdown(f"""
                <div class="dashboard-box">
                    <h4 style="❌ margin-top: 0;"> Features Eliminadas</h4>
                    <ul style="list-style-type: none; padding-left: 0; line-height: 2;">
                """, unsafe_allow_html=True)
                for col in sorted(columnas_eliminadas):
                    st.markdown(f"""
                        <li style="color: #000000;">  <!-- Punto negro -->
                        <strong style="color: {colors_si_no[1]};">❌{col}</strong>  <!-- Texto rojo -->
                        </li>
                    """, unsafe_allow_html=True)
                st.markdown(f"""
                        <li style="margin-top: 10px; color: #6c757d; font-size: 12px;">Total: {len(columnas_eliminadas)} columnas eliminadas por no ser relevantes</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="dashboard-box">
                    <h4 style="color: #28a745; margin-top: 0;">✅ Todas las columnas se mantuvieron</h4>
                    <p style="color: #6c757d;">No se eliminaron columnas durante el preprocesamiento</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.divider()
        
       
        # Estructura final del dataset
        # Fila 5
        st.markdown('<h3 class="subsection-title">📊 Estructura Final del Dataset</h3>', unsafe_allow_html=True)
        
        # Mostrar información de tipos y estadísticas
        col1, col2 = st.columns(2)
        
        with col1:
            with st.container(border=True):               
                st.markdown("**📋 Tipos de datos**")
                tipo_df = df_visu.dtypes.reset_index()
                tipo_df.columns = ['Feature', 'Tipo']
                
                tipo_df['_index'] = tipo_df.index
                
                styled_df = tipo_df.style.apply(lambda x: 
                    ['background-color: ' + colores_alternas[0] if i % 2 == 0 
                     else 'background-color: ' + colores_alternas[1] 
                     for i in range(len(x))], axis=0)
                st.dataframe(styled_df, use_container_width=True)

                
                
                #st.dataframe(tipo_df, use_container_width=True)
                
        with col2:
            with st.container(border=True):          
                st.markdown("**📈 Estadísticas descriptivas**")
                tipo_df = df_visu.describe()
                
                # ✅ Aplicar estilos directamente sin crear columna _index
                styled_df = tipo_df.style.apply(lambda x: 
                    ['background-color: ' + (colores_alternas[0] if i % 2 == 0 else colores_alternas[1])
                     for i in range(len(x))], axis=0)
                
                st.dataframe(styled_df, use_container_width=True)
                            
                #st.dataframe(df_visu.describe(), use_container_width=True)

        # Mostrar información básica de resultados
        if 'dimensiones_limpios' in results:
            st.markdown(f"""
            <div class="dashboard-box">
                <h4 style="color: #1a7a3a;">📊 Resumen de la limpieza (desde resultados)</h4>
                <ul style="list-style-type: none; padding-left: 0; line-height: 2;">
                    <li>📋 Registros finales: {results['dimensiones_limpios'][0]}</li>
                    <li>📊 Features finales: {results['dimensiones_limpios'][1]}</li>
                    <li>🗑️ Features eliminadas: {df_raw.shape[1] - results['dimensiones_limpios'][1]}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        st.divider()
        # ============================================
    # OPCIÓN 2: % OUTLIERS POR FEATURE (barras)
    # ============================================
    
    st.subheader("📊 Porcentaje de Outliers por Feature")
    top_n=20
    
    df_top = outliers_df_summary.head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(df_top) * 0.3)))
    
    colors = [colors_si_no[1] if pct > 5 else colors_si_no[0] for pct in df_top['pct']]
    
    
    bars = ax.barh(df_top.index, df_top['pct'], color=colors, edgecolor='black')
    
    ax.set_xlabel('% Outliers', fontsize=12)
    ax.set_title(f'Top {top_n} Features con más Outliers (método IQR)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    for bar, pct in zip(bars, df_top['pct']):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
               f'{pct:.1f}%', va='center', fontsize=8)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# TAB3 Exploracion Edad/Sexo  
with tab3:
    # Después de cargar results:
    st.write("Estructura de results:")
    st.write(results.keys())
    df_sexo_edad = results['df_sexo_edad']
    contingencia = results['contingencia']
    stats_sexo = results['stats_sexo']
    boxplot_data = results['boxplot_data']
    bins = results['bins']
    labels = results['labels']
    sexos = results['sexos']
    
    # ============================================
    # TÍTULO DE LA TAB
    # ============================================
    st.header("📊 Exploración: Edad y Sexo")
    
    # ============================================
    # MÉTRICAS RÁPIDAS (arriba del todo)
    # ============================================
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total pacientes", len(df_sexo_edad))
    with col2:
        st.metric("Edad promedio", f"{df_sexo_edad['Edad'].mean():.1f} años")
    with col3:
        st.metric("Edad mínima", f"{df_sexo_edad['Edad'].min():.0f} años")
    with col4:
        st.metric("Edad máxima", f"{df_sexo_edad['Edad'].max():.0f} años")
    
    st.divider()
    
    # ============================================
    # LAYOUT PRINCIPAL: CONTROLES (1/3) + GRÁFICOS (2/3)
    # ============================================
    col_controls, col_plots = st.columns([1, 2])  # 1/3 y 2/3
    
    # ============================================
    # COLUMNA IZQUIERDA: CONTROLES
    # ============================================
    with col_controls:
        st.subheader("🎛️ Controles")
        
        # Selección de tipo de gráfico
        tipo_grafico = st.radio(
            "📊 Tipo de gráfico:",
            ["Histograma", "Boxplot"],
            key="tipo_grafico"
        )
        
        st.divider()
        
        # Selección de sexo
        sexo_seleccionado = st.radio(
            "👥 Filtrar por sexo:",
            ["Ambos", "Sexo 1", "Sexo 2"],
            key="sexo_seleccionado"
        )
        
        st.divider()
        
        # Mostrar resumen de filtros aplicados
        st.caption("🔍 **Filtros aplicados:**")
        st.caption(f"- Gráfico: {tipo_grafico}")
        st.caption(f"- Sexo: {sexo_seleccionado}")
        
        # Mostrar estadísticas rápidas del filtro
        if sexo_seleccionado == "Ambos":
            df_filtro = df_sexo_edad
        elif sexo_seleccionado == "Sexo 1":
            df_filtro = df_sexo_edad[df_sexo_edad['Sexo'] == 1]
        else:
            df_filtro = df_sexo_edad[df_sexo_edad['Sexo'] == 2]
        
        st.divider()
        st.caption(f"📊 **n = {len(df_filtro)}** pacientes")
        if len(df_filtro) > 0:
            st.caption(f"Edad: {df_filtro['Edad'].min():.0f} - {df_filtro['Edad'].max():.0f} años")
            st.caption(f"Media: {df_filtro['Edad'].mean():.1f} años")
    
    # ============================================
    # COLUMNA DERECHA: GRÁFICOS
    # ============================================
    with col_plots:
        # Convertir selección a filtro
        if sexo_seleccionado == "Ambos":
            sexo_filter = None
            titulo_sexo = "Ambos sexos"
        elif sexo_seleccionado == "Sexo 1":
            sexo_filter = 1
            titulo_sexo = "Sexo 1"
        else:
            sexo_filter = 2
            titulo_sexo = "Sexo 2"
        
        # Filtrar datos
        if sexo_filter is not None:
            df_filtrado = df_sexo_edad[df_sexo_edad['Sexo'] == sexo_filter]
        else:
            df_filtrado = df_sexo_edad
        
        
        
        
        # ============================================
        # GRÁFICO PRINCIPAL
        # ============================================
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if tipo_grafico == "Histograma":
            # Definir colores por sexo 
            #colors_sexo = {1: '#7FE817', 2: '#FFFF00'}
            
            if sexo_filter is not None:
                # Si hay filtro, histograma simple
                ax.hist(df_filtrado['Edad'], bins=bins, edgecolor='black', 
                        color=colors_sexo[sexo_filter], alpha=0.7)
                ax.set_title(f'Histograma de Edad - {titulo_sexo}', fontsize=14, fontweight='bold')
                ax.set_xlabel('Edad', fontsize=12, fontweight='bold')
                ax.set_ylabel('Frecuencia', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                # Añadir media y mediana
                if len(df_filtrado) > 0:
                    media = df_filtrado['Edad'].mean()
                    mediana = df_filtrado['Edad'].median()
                    ax.axvline(media, color='red', linestyle='--', linewidth=2, label=f'Media: {media:.1f}')
                    ax.axvline(mediana, color='blue', linestyle='-.', linewidth=2, label=f'Mediana: {mediana:.1f}')
                    ax.legend()
            
            else:
                # ============================================
                # BARRAS APILADAS (CUANDO NO HAY FILTRO)
                # ============================================
                
                # Crear tabla de contingencia para el gráfico
                contingencia_plot = pd.crosstab(
                    pd.cut(df_filtrado['Edad'], bins=bins, labels=labels, right=False),
                    df_filtrado['Sexo']
                )
                
                # Obtener los sexos únicos
                sexos_plot = contingencia_plot.columns.tolist()
                
                # Crear barras apiladas
                bottom = np.zeros(len(contingencia_plot))
                for i, sexo in enumerate(sexos_plot):
                    ax.bar(contingencia_plot.index, contingencia_plot[sexo],
                           bottom=bottom, label=f'Sexo {sexo}', 
                           color=colors_sexo[sexo],
                           edgecolor='white', linewidth=1, alpha=0.8)
                    bottom += contingencia_plot[sexo].values
                
                # Personalizar
                ax.set_xlabel('Rango de Edad', fontsize=12, fontweight='bold')
                ax.set_ylabel('Número de personas', fontsize=12, fontweight='bold')
                ax.set_title('Distribución de Sexo por Rango de Edad (Barras Apiladas)',
                             fontsize=14, fontweight='bold')
                ax.legend(title='Sexo', bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, alpha=0.3, axis='y')
                
                # Añadir valores de cada parte
                bottom = np.zeros(len(contingencia_plot))
                for i, sexo in enumerate(sexos_plot):
                    valores = contingencia_plot[sexo].values
                    for j, valor in enumerate(valores):
                        if valor > 0:
                            y_pos = bottom[j] + valor / 2
                            ax.text(j, y_pos, str(int(valor)),
                                    ha='center', va='center',
                                    fontweight='bold', fontsize=10, color='black')
                    bottom += valores
                
                # Añadir totales en las barras
                for i, (idx, row) in enumerate(contingencia_plot.iterrows()):
                    total = row.sum()
                    ax.text(i, total + 1, f' n={int(total)}',
                            ha='center', va='bottom', fontweight='bold', fontsize=9, color='#333')
                
                # Rotar etiquetas del eje x si son muchas
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        else:  # Boxplot
            
            #colors_sexo = {1: '#7FE817', 2: '#FFFF00'}
            
            if sexo_filter is not None:
                datos_box = df_filtrado['Edad'].tolist()
                if len(datos_box) > 0:
                    box = ax.boxplot(datos_box, patch_artist=True, 
                                    widths=0.6,
                                    medianprops=dict(color='black', linewidth=2))
                    box['boxes'][0].set_facecolor(colors_sexo[sexo_filter])
                    box['boxes'][0].set_alpha(0.7)
                    ax.set_xticklabels([f'Sexo {sexo_filter}'])
                ax.set_title(f'Boxplot de Edad - {titulo_sexo}', fontsize=14, fontweight='bold')
            else:
                datos_box = [df_filtrado[df_filtrado['Sexo'] == sexo]['Edad'].tolist() for sexo in sexos]
                colores_box = [colors_sexo[sexo] for sexo in sexos]
                
                datos_box_con_datos = [d for d in datos_box if len(d) > 0]
                if len(datos_box_con_datos) > 0:
                    box = ax.boxplot(datos_box_con_datos, patch_artist=True,
                                    labels=[f'Sexo {s}' for s in sexos if len(df_filtrado[df_filtrado['Sexo'] == s]) > 0],
                                    widths=0.6,
                                    medianprops=dict(color='black', linewidth=2))
                    
                    for patch, color in zip(box['boxes'], colores_box):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
                ax.set_title('Boxplot de Edad por Sexo', fontsize=14, fontweight='bold')
            
            ax.set_xlabel('Sexo', fontsize=12, fontweight='bold')
            ax.set_ylabel('Edad', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
       
        # Tabla de contingencia(por edad) (debajo del gráfico columnas apiladas)
        # ============================================
        st.subheader("📋 Distribución por Rangos de Edad")


        

        # Filtrar contingencia si es necesario
        if sexo_filter is not None:
            contingencia_filtrada = contingencia[[sexo_filter]]
            contingencia_filtrada['Total'] = contingencia_filtrada[sexo_filter]
        else:
            contingencia_filtrada = contingencia.copy()
            contingencia_filtrada['Total'] = contingencia.sum(axis=1)
        
        # Crear función de estilo
        def apply_styles(row):
            idx = row.name
            # Obtener el índice numérico real (si el índice no es numérico, usar posición)
            try:
                pos = contingencia_filtrada.index.get_loc(idx)
            except:
                pos = 0
            
            bg_color = colores_alternas[0] if pos % 2 == 0 else colores_alternas[1]
            
            styles = []
            for col in row.index:
                style = f'background-color: {bg_color}; text-align: center;'
                if col in [1, 2] and col in colors_sexo:
                    style = f'background-color: {colors_sexo[col]}; color: black; font-weight: bold; text-align: center;'
                elif col == 'Total':
                    style = f'background-color: {bg_color}; text-align: center;'
                styles.append(style)
            return styles
        
        # Aplicar estilos
        styled_df = contingencia_filtrada.style.apply(apply_styles, axis=1)
        
        # Formatear números
        styled_df = styled_df.format(lambda x: f'{int(x):d}' if isinstance(x, (int, float)) and not pd.isna(x) else x)
        
        # Mostrar tabla
        st.dataframe(
            styled_df,
            use_container_width=True,
            column_config={
                "Total": st.column_config.NumberColumn("Total", format="%d")
            }
        )
        
        
        
   
    # Estadisticas descriptiva(2 filas)
    
    st.divider()
    
    col_stats = st.columns(1)[0]
    
    with col_stats:
        st.subheader("📊 Estadísticas Descriptivas por Sexo")
        
        # Filtrar estadísticas
        if sexo_filter is not None:
            stats_filtrada = stats_sexo.loc[[sexo_filter]]
            df=stats_filtrada.T
           
        else:
            df=stats_sexo
            
        # Aplicar colores alternos con pandas Styler
        styled_df = df.style.apply(
            lambda x: ['background-color: ' + (colors_sexo[1] if i % 2 == 0 else colors_sexo[2]) 
                       for i in range(len(x))], 
            axis=0
        )
        
        # Mostrar tabla con formato
        st.dataframe(
            styled_df,
            use_container_width=True,
            column_config={
                "Total": st.column_config.NumberColumn("Total", format="%d")
            }
        )
        
        
    st.divider()
    
    # Estadisticas de rango de edad: numeros globales
    col1=st.columns(1)[0]
    
    with col1:
        st.subheader("🔍 Análisis de Balanceo por Sexo")
        # Balanceo Global por Sexo
        if sexo_filter is None:
  
            sexo_counts = df_sexo_edad['Sexo'].value_counts()
            sexo_pct = (sexo_counts / len(df_sexo_edad) * 100).round(1)
            
            st.write("**1. Balanceo Global por Sexo:**")
            col_sex1, col_sex2 = st.columns(2)
            with col_sex1:
                color_sex1 = colors_sexo.get(1, '#000000')
                st.metric(
                    "Sexo 1", 
                    f"{sexo_counts[1]} ({sexo_pct[1]:.1f}%)",
                    delta=None
                )
                # Añadir color mediante HTML (no afecta a st.metric directamente)
                st.markdown(f'<style>div[data-testid="metric-container"]:first-child {{ color: {color_sex1}; }}</style>', unsafe_allow_html=True)
            with col_sex2:
                color_sex2 = colors_sexo.get(2, '#000000')
                st.metric(
                    "Sexo 2", 
                    f"{sexo_counts[2]} ({sexo_pct[2]:.1f}%)",
                    delta=None
                )
                st.markdown(f'<style>div[data-testid="metric-container"]:last-child {{ color: {color_sex2}; }}</style>', unsafe_allow_html=True)
            
            diff_global = abs(sexo_pct[1] - sexo_pct[2])
            if diff_global < 10:
                st.success(f"✅ Globalmente balanceado: diferencia de {diff_global:.1f}%")
            else:
                st.warning(f"⚠️ Globalmente desbalanceado: diferencia de {diff_global:.1f}%")
         
    st.divider()
        
    
    # Desbalanceo por rango: colores- intensidad
    col2=st.columns(1)[0]       
    with col2:
        # ============================================
        # 3. GRÁFICO DE DESBALANCEO POR RANGO
        # ============================================
        st.write("**3. Visualización del Desbalanceo por Rango:**")
        
        # ✅ CREAR df_balanceo ANTES DE USARLO
        # Calcular diferencias por rango de edad
        df_balanceo = df_sexo_edad.groupby('Rango_Edad').apply(
            lambda x: pd.Series({
                'Total': len(x),
                'Sexo1': (x['Sexo'] == 1).sum(),
                'Sexo2': (x['Sexo'] == 2).sum(),
                'Diferencia_%': abs((x['Sexo'] == 1).sum() - (x['Sexo'] == 2).sum()) / len(x) * 100
            })
        ).reset_index()
        
        # Crear gráfico de barras para diferencias
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        
        # Colores según nivel de desbalanceo
        colores_balanceo = {
            "✅ Balanceado": '#28a745',
            "⚠️ Moderado": '#ffc107',
            "❌ Desbalanceado": '#dc3545'
        }
        
        
        
        colors_edad_lista = [
            '#FFF5F5',  # 0: muy claro (casi blanco)
            '#FFE0E0',  # 1: claro
            '#FFCCCC',  # 2: medio claro
            '#FF9999',  # 3: medio
            '#FF6666',  # 4: medio oscuro
            '#FF3333',  # 5: oscuro
            '#CC0000'   # 6: muy oscuro (rojo intenso)
        ]
        
       
        # Asignar colores según el nivel de desbalanceo
        max_diff = df_balanceo['Diferencia_%'].max()
        if max_diff > 0:
            normalized_diff = df_balanceo['Diferencia_%'] / max_diff
            color_indices = (normalized_diff * 6).astype(int).clip(0, 6)
            colors_bar = [colors_edad_lista[idx] for idx in color_indices]
        else:
            colors_bar = [colors_edad_lista[0]] * len(df_balanceo)
        
        bars = ax2.bar(df_balanceo['Rango_Edad'], df_balanceo['Diferencia_%'], 
                      color=colors_bar, edgecolor='black', linewidth=1, alpha=0.8)
        
        # Línea de referencia para balanceado (<10%)
        ax2.axhline(y=10, color='red', linestyle='--', linewidth=1.5, 
                   label='Límite balanceado (10%)', alpha=0.7)
        
        # Añadir valores y n en las barras
        for i, (idx, row) in enumerate(df_balanceo.iterrows()):
            ax2.text(i, row['Diferencia_%'] + 1, 
                    f"{row['Diferencia_%']:.1f}%\nn={int(row['Total'])}",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax2.set_xlabel('Rango de Edad', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Diferencia entre sexos (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Desbalanceo por Rango de Edad (diferencia absoluta entre sexos)',
                     fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim(0, max(df_balanceo['Diferencia_%']) + 5)
        
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)
     
    
    # Nueva fila
    col3=st.columns(1)[0]
    with col3:
        # ============================================
        # 4. RESUMEN DEL DESBALANCEO POR RANGOS
        # ============================================
        st.divider()
        st.write("**4. Resumen del Desbalanceo por Rangos:**")
        
        col_res1, col_res2, col_res3 = st.columns(3)
        
        # Calcular porcentajes por rango de edad
        contingencia_pct = contingencia.div(contingencia.sum(axis=1), axis=0) * 100
        contingencia_pct = contingencia_pct.round(1)
        
        # Calcular diferencia por rango
        diff_por_rango = {}
        for idx in contingencia.index:
            if 1 in contingencia.columns and 2 in contingencia.columns:
                diff = abs(contingencia_pct.loc[idx, 1] - contingencia_pct.loc[idx, 2])
                diff_por_rango[idx] = diff
        
        # Crear DataFrame para visualización
        df_balanceo = pd.DataFrame({
            'Rango': list(diff_por_rango.keys()),
            'Diferencia_%': list(diff_por_rango.values()),
            'Sexo_1_%': [contingencia_pct.loc[idx, 1] for idx in diff_por_rango.keys()],
            'Sexo_2_%': [contingencia_pct.loc[idx, 2] for idx in diff_por_rango.keys()],
            'Total': [contingencia.loc[idx].sum() for idx in diff_por_rango.keys()]
        })
        
        # Clasificar balanceo por rango
        def clasificar_balanceo(diff):
            if diff < 10:
                return "✅ Balanceado"
            elif diff < 20:
                return "⚠️ Moderado"
            else:
                return "❌ Desbalanceado"
        
        df_balanceo['Estado'] = df_balanceo['Diferencia_%'].apply(clasificar_balanceo)
        
        with col_res1:
            n_balanceados = len(df_balanceo[df_balanceo['Estado'] == "✅ Balanceado"])
            pct_balanceados = round(n_balanceados / len(df_balanceo) * 100, 1)
            st.metric("Rangos Balanceados", 
                     f"{n_balanceados}/{len(df_balanceo)}",
                     delta=f"{pct_balanceados}%")
        
        with col_res2:
            n_moderados = len(df_balanceo[df_balanceo['Estado'] == "⚠️ Moderado"])
            st.metric("Rangos Moderados", f"{n_moderados}")
        
        with col_res3:
            n_desbalanceados = len(df_balanceo[df_balanceo['Estado'] == "❌ Desbalanceado"])
            st.metric("Rangos Desbalanceados", f"{n_desbalanceados}")
        
        # Identificar los rangos más problemáticos
        if n_desbalanceados > 0:
            peores_rangos = df_balanceo[df_balanceo['Estado'] == "❌ Desbalanceado"].sort_values('Diferencia_%', ascending=False)
            st.warning(f"⚠️ **Rangos con mayor desbalanceo:** {', '.join(peores_rangos['Rango'].tolist())}")
            
            # Mostrar detalle del peor rango
            peor = peores_rangos.iloc[0]
            st.info(f"📌 **Peor caso:** Rango {peor['Rango']} - "
                   f"Sexo 1: {peor['Sexo_1_%']:.1f}% vs Sexo 2: {peor['Sexo_2_%']:.1f}% "
                   f"(diferencia: {peor['Diferencia_%']:.1f}%)")
    
        #else:
         #   st.info("📌 Selecciona 'Ambos' en controles para ver el análisis de balanceo completo")

# TAB4:DIAGNÓSTICOS
with tab4:

    st.title("📊 Análisis de Consistencia entre Diagnósticos")
    df_outliers = results['df_outliers']
    df_frecuencias = results['df_frecuencias']
    df_diagnostico_unico = results['df_diagnostico_unico']
    estadisticas = results['estadisticas']
    st.subheader("📈 Resumen General")

    # métricas globales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Pacientes", estadisticas['total_pacientes'])
    with col2:
        st.metric("Patrones Únicos", estadisticas['n_patrones_unicos'])
    with col3:
        st.metric("Diagnósticos Emitidos", results['total_diagnosticos_emitidos'])
    with col4:
        st.metric("Diagnósticos Únicos", len(df_diagnostico_unico))
    
    st.divider()
    

    
    # Diagnósticos: tabla +pie-chart
    st.subheader("📊 Diagnósticos Únicos Emitidos")
    
    col1,col2=st.columns(2)
    
    with col1:
        # Asumiendo que la columna 'Diagnóstico' contiene ED1, ED2, ED3
        def color_diagnostico(row):
            diag = row['Diagnóstico']
            if diag in colors_diag:
                return ['background-color: ' + colors_diag[diag]] * len(row)
            else:
                return [''] * len(row)
            
        # Redondear a 2 decimales
        df_diagnostico_unico['Frecuencia'] = df_diagnostico_unico['Frecuencia'].round(2)
        df_diagnostico_unico['Porcentaje (%)'] = df_diagnostico_unico['Porcentaje (%)'].round(2)
        
        # Si hay columna de porcentaje, formatear como string con %
        df_diagnostico_unico['Porcentaje (%)'] = df_diagnostico_unico['Porcentaje (%)'].apply(lambda x: f"{x:.2f}")
        
        # Aplicar estilo y mostrar
        styled_df = df_diagnostico_unico.style.apply(color_diagnostico, axis=1)
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
   
    with col2:
        #vacio
        st.subheader("")
   
    st.divider() 
    
    col1=st.columns(1)[0]
    with col1:
        
        # los diferentes patrones
        def color_etiquetas(valor):
            """Convierte una tupla de etiquetas en HTML con colores"""
            if not valor or len(valor) == 0:
                return ''
            
            if isinstance(valor, tuple):
                etiquetas = valor
            else:
                etiquetas = str(valor).split()
            
            spans = []
            for etiqueta in etiquetas:
                etiqueta = etiqueta.strip()
                if etiqueta in colors_diag:
                    bg_color = colors_diag[etiqueta]
                    spans.append(f'<span style="background-color: {bg_color}; color: black; font-weight: bold; border-radius: 12px; padding: 2px 8px; margin: 1px; display: inline-block; font-size: 12px;">{etiqueta}</span>')
                else:
                    spans.append(f'<span style="background-color: #f0f0f0; color: black; border-radius: 12px; padding: 2px 8px; margin: 1px; display: inline-block; font-size: 12px;">{etiqueta}</span>')
            
            return ' '.join(spans)
        
        # Crear columna HTML
        df_frecuencias['Patrón_HTML'] = df_frecuencias['Patrón'].apply(color_etiquetas)
        
        # Calcular proporción ED1:ED2:ED3
        def calcular_proporcion(patron):
            """Calcula la proporción de ED1, ED2, ED3 en el patrón"""
            if not patron or len(patron) == 0:
                return ''
            
            if isinstance(patron, tuple):
                etiquetas = patron
            else:
                etiquetas = str(patron).split()
            
            # Contar cada diagnóstico
            count_ed1 = sum(1 for e in etiquetas if e == 'ED1')
            count_ed2 = sum(1 for e in etiquetas if e == 'ED2')
            count_ed3 = sum(1 for e in etiquetas if e == 'ED3')
            
            total = len(etiquetas)
            if total == 0:
                return ''
            
            # Formatear como "ED1:ED2:ED3"
            return f'{count_ed1}:{count_ed2}:{count_ed3}'
        
        df_frecuencias['Proporción'] = df_frecuencias['Patrón'].apply(calcular_proporcion)
        
        # Construir tabla HTML manual
        html_tabla = '<table style="width:100%; border-collapse: collapse; font-family: sans-serif;">'
        html_tabla += '<tr>'
        html_tabla += '<th style="border:1px solid #ddd; padding:8px; background-color:#f0f2f6; text-align:center;">Patrón</th>'
        html_tabla += '<th style="border:1px solid #ddd; padding:8px; background-color:#f0f2f6; text-align:center;">Frecuencia</th>'
        html_tabla += '<th style="border:1px solid #ddd; padding:8px; background-color:#f0f2f6; text-align:center;">Porcentaje (%)</th>'
        html_tabla += '<th style="border:1px solid #ddd; padding:8px; background-color:#f0f2f6; text-align:center;">Proporción<br>(ED1:ED2:ED3)</th>'  # 🔥 NUEVA COLUMNA
        html_tabla += '</tr>'
        
        for _, row in df_frecuencias.iterrows():
            html_tabla += '<tr>'
            html_tabla += f'<td style="border:1px solid #ddd; padding:8px;">{row["Patrón_HTML"]}</td>'
            html_tabla += f'<td style="border:1px solid #ddd; padding:8px; text-align:center;">{row["Frecuencia"]}</td>'
            html_tabla += f'<td style="border:1px solid #ddd; padding:8px; text-align:center;">{row["Porcentaje(pacientes) (%)"]:.2f}</td>'
            html_tabla += f'<td style="border:1px solid #ddd; padding:8px; text-align:center; font-weight: bold;">{row["Proporción"]}</td>'  # 🔥 NUEVA COLUMNA
            html_tabla += '</tr>'
        
        html_tabla += '</table>'
        
        st.markdown(html_tabla, unsafe_allow_html=True)
        
        st. divider()
    col1, col2 = st.columns(2)
    
    with col1:
        #pie de volores unicos diagnosticos
        # Ordenar y mapear
        df_ordenado = df_diagnostico_unico.sort_values('Frecuencia', ascending=False)
        orden = df_ordenado['Diagnóstico'].tolist()
        
        fig = px.pie(
            df_ordenado,
            values='Frecuencia',
            names='Diagnóstico',
            title='Distribución de Diagnósticos Emitidos',
            color='Diagnóstico',
            color_discrete_map=colors_diag,
            category_orders={'Diagnóstico': orden}
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=350, legend_traceorder='normal')
        st.plotly_chart(fig, use_container_width=True)
         
    with col2:
        # Calcular proporciones (si no están en el PKL, las calculamos)
        if 'proporcion' in df_outliers.columns:
            proporcion_counts = df_outliers['proporcion'].value_counts()
            proporcion_pct = (proporcion_counts / len(df_outliers) * 100).round(2)
            
            df_proporciones = pd.DataFrame({
                'Proporción': proporcion_counts.index,
                'Conteo': proporcion_counts.values,
                'Porcentaje (%)': proporcion_pct.values
            })
            
        #pie de proporciones
        fig = px.pie(
            df_proporciones,
            values='Conteo',
            names='Proporción',
            title='Distribución de Proporciones en los Patrones',
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
        
    st.divider()    
 
    #Barras apiladas: patrones y su frecuencias

    st.subheader("📊 Patrones Encontrados y su Composición")
    col1=st.columns(1)[0]

    with col1:
        #top_n = st.slider("Número de patrones a mostrar:", 3, 15, 6)
     
        top_n=6
        # Tomar top N patrones
        df_top = df_frecuencias.head(top_n).copy()
        
        # Extraer composición de cada patrón
        def extraer_composicion(patron):
            counts = {'ED1': 0, 'ED2': 0, 'ED3': 0}
            for val in patron:
                if val in counts:
                    counts[val] += 1
            return counts
        
        composiciones = df_top['Patrón'].apply(extraer_composicion)
        df_composicion = pd.DataFrame(composiciones.tolist(), index=df_top.index)
        
        # Unir con los datos originales
        df_top_completo = pd.concat([df_top, df_composicion], axis=1)
        
        
        # Cada segmento = (conteo_del_diagnostico / 5) * frecuencia
        # Así la barra total = frecuencia (ej: 85)
        for dx in ['ED1', 'ED2', 'ED3']:
            df_top_completo[f'{dx}_altura'] = (df_top_completo[dx] / 5) * df_top_completo['Frecuencia']
        
        # Crear gráfico de barras apiladas
        fig = go.Figure()
        
        colores = colors_diag
        etiquetas = {'ED1': 'ED1', 'ED2': 'ED2', 'ED3': 'ED3'}
        
        for dx, color in colores.items():
            fig.add_trace(go.Bar(
                name=etiquetas[dx],
                x=df_top_completo['Patrón'].astype(str),
                y=df_top_completo[f'{dx}_altura'],
                marker_color=color,
                text=df_top_completo[f'{dx}_altura'].round(1),
                textposition='inside',
                hovertemplate=f'%{{x}}<br>{dx}: %{{customdata:.1f}} pacientes<extra></extra>',
                customdata=df_top_completo[f'{dx}_altura']
            ))
        
        # Configurar diseño
        fig.update_layout(
            barmode='stack',
            title=f'Los Patrones y su Composición',
            xaxis_title='Patrón',
            yaxis_title='Número de Pacientes',
            legend_title='Diagnóstico',
            height=500,
            xaxis={'tickangle': -30},
            yaxis=dict(range=[0, df_top_completo['Frecuencia'].max() * 1.15])
        )
        
        # Añadir total (frecuencia) en la parte superior de cada barra
        for i, row in df_top_completo.iterrows():
            total = row['Frecuencia']
           
            pct = row['Porcentaje(pacientes) (%)']
            patron_str = str(row['Patrón'])
            fig.add_annotation(
                x=patron_str,
                y=total + 8,
                text=f"<b>Total:</b><br>{total} ({pct}%)",
                showarrow=False,
                font=dict(size=11, color='black'),
                bgcolor='white',
                bordercolor='black',
                borderwidth=1
            )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
    
    

    
        # Filtro interactivos de patrones por su composicion

    
        st.subheader("🔍 Filtrar Patrones por Proporción")

        if 'proporcion' in df_outliers.columns:
            proporcion_seleccionada = st.selectbox(
                "Selecciona una proporción:",
                ['Todas'] + sorted(df_outliers['proporcion'].unique().tolist())
            )
            
            if proporcion_seleccionada != 'Todas':
                df_filtrado = df_outliers[df_outliers['proporcion'] == proporcion_seleccionada]
                df_filtrado_patrones = df_filtrado['patron_ordenado'].value_counts().reset_index()
                df_filtrado_patrones.columns = ['Patrón', 'Frecuencia']
                
                # ✅ Aplicar colores a los patrones
                def color_etiquetas_patron(patron):
                    """Convierte un patrón (tupla o string) en HTML con colores"""
                    if not patron:
                        return ''
                    
                    # Si es string, convertirlo a lista
                    if isinstance(patron, str):
                        etiquetas = patron.split()
                    else:
                        etiquetas = patron
                    
                    spans = []
                    for etiqueta in etiquetas:
                        etiqueta = etiqueta.strip()
                        if etiqueta in colors_diag:
                            bg_color = colors_diag[etiqueta]
                            spans.append(f'<span style="background-color: {bg_color}; color: black; font-weight: bold; border-radius: 12px; padding: 2px 8px; margin: 1px; display: inline-block; font-size: 12px;">{etiqueta}</span>')
                        else:
                            spans.append(f'<span style="background-color: #f0f0f0; color: black; border-radius: 12px; padding: 2px 8px; margin: 1px; display: inline-block; font-size: 12px;">{etiqueta}</span>')
                    
                    return ' '.join(spans)
                
                # Crear columna HTML
                df_filtrado_patrones['Patrón_HTML'] = df_filtrado_patrones['Patrón'].apply(color_etiquetas_patron)
                
                # Mostrar tabla con HTML
                html_tabla = '<table style="width:100%; border-collapse: collapse; font-family: sans-serif;">'
                html_tabla += '<tr>'
                html_tabla += '<th style="border:1px solid #ddd; padding:8px; background-color:#f0f2f6; text-align:center;">Patrón</th>'
                html_tabla += '<th style="border:1px solid #ddd; padding:8px; background-color:#f0f2f6; text-align:center;">Frecuencia</th>'
                html_tabla += '</tr>'
                
                for _, row in df_filtrado_patrones.iterrows():
                    html_tabla += '<tr>'
                    html_tabla += f'<td style="border:1px solid #ddd; padding:8px;">{row["Patrón_HTML"]}</td>'
                    html_tabla += f'<td style="border:1px solid #ddd; padding:8px; text-align:center;">{row["Frecuencia"]}</td>'
                    html_tabla += '</tr>'
                
                html_tabla += '</table>'
                
                st.markdown(html_tabla, unsafe_allow_html=True)
                st.caption(f"Mostrando {len(df_filtrado_patrones)} patrones con proporción {proporcion_seleccionada}")

            else:
                st.info("Selecciona una proporción para ver los patrones filtrados")
        
        st.divider()

    # ============================================
    # Número de diagnósticos diferentes para un paciente 
    # ============================================
    st.subheader("Distribución de Número de Diagnósticos por Paciente")

    # Calcular cuántos diagnósticos únicos tiene cada paciente
    # Los diagnósticos son D1-D5
    diagnostic_cols = ['D1', 'D2', 'D3', 'D4', 'D5']
    df = results['df_outliers']
    
    # Contar diagnósticos únicos por paciente
    df['num_diagnosticos_unicos'] = df[diagnostic_cols].apply(
        lambda row: len(set(row)), axis=1
    )
    
    # Distribución
    counts = df['num_diagnosticos_unicos'].value_counts().sort_index()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        df_unique = pd.DataFrame({
            'Número de diagnósticos diferentes': counts.index,
            'Pacientes': counts.values
        })
        
        # ✅ Aplicar colores alternos con pandas Styler
        styled_df = df_unique.style.apply(
            lambda x: ['background-color: ' + (colores_alternas[0] if i % 2 == 0 else colores_alternas[1]) 
                       for i in range(len(x))], 
            axis=0
        )

        st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))

        # ✅ Asignar colores según el número de diagnósticos
        # Si count = 1 → verde (colors_si_no[0]), si count > 1 → rojo (colors_si_no[1])
        colores_barras = [colors_si_no[0] if c == 1 else colors_si_no[1] for c in counts.index]
        
        ax.bar(counts.index, counts.values, color=colores_barras, edgecolor='black', linewidth=1)
        
        ax.set_title('Número de Diagnósticos Diferentes por Paciente', fontsize=14, fontweight='bold')
        ax.set_xlabel('Número de diagnósticos únicos', fontsize=12)
        ax.set_ylabel('Número de pacientes', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Añadir valores sobre las barras
        for key, val in counts.items():
            ax.text(key, val + 0.5, str(val), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
            
    
    
    st.divider()
    
    
    # Distribucion patrones comun->muy raro 
    col1=st.columns(1)[0]
    st.subheader("Distribución y nivel de rareza de los patrones")
    
    
    
    
    # Crear figura con dos subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ============================================
    # GRÁFICO 1: Distribución de grupos de patrón
    # ============================================
    ax1 = axes[0]
    
    # Contar frecuencia de cada grupo
    grupos_counts = df_outliers['grupo_patron'].value_counts()
    
    # Ordenar según rareza (de más común a más raro)
    orden_grupos = ['comun', 'poco comun', 'raro', 'muy raro']
    grupos_counts = grupos_counts.reindex(orden_grupos)
    
    # Crear barras con colores del árbol
    bars = ax1.bar(
        grupos_counts.index, 
        grupos_counts.values,
        color=[colors_rareza.get(x, '#808080') for x in grupos_counts.index],
        edgecolor='black', 
        alpha=0.8,
        linewidth=1.5
    )
    
    # Configurar título y ejes
    ax1.set_title('Distribución de Grupos de Patrón', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Grupo', fontsize=12)
    ax1.set_ylabel('Número de Pacientes', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Agregar etiquetas con conteo y porcentaje
    for bar, count in zip(bars, grupos_counts.values):
        pct = (count / len(df_outliers)) * 100
        ax1.text(
            bar.get_x() + bar.get_width()/2, 
            bar.get_height() + 1,
            f'{count} ({pct:.1f}%)', 
            ha='center', 
            va='bottom', 
            fontsize=10,
            fontweight='bold'
        )
    
    # ============================================
    # GRÁFICO 2: Outliers vs No Outliers
    # ============================================
    ax2 = axes[1]
    
    # Contar outliers
    outlier_counts = df_outliers['outlier_patron'].value_counts()
    
    # Preparar datos
    tipos = ['No Outlier', 'Outlier']
    valores = [
        outlier_counts.get(False, 0),  # No Outlier
        outlier_counts.get(True, 0)    # Outlier
    ]
    colores_outliers = [colors_si_no[0], colors_si_no[1]]
    
    # Crear barras
    bars2 = ax2.bar(
        tipos, 
        valores,
        color=colores_outliers, 
        edgecolor='black', 
        alpha=0.8,
        linewidth=1.5
    )
    
    # Configurar título y ejes
    ax2.set_title('Outliers vs No Outliers', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Tipo', fontsize=12)
    ax2.set_ylabel('Número de Pacientes', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Agregar etiquetas con conteo y porcentaje
    for bar, count in zip(bars2, valores):
        pct = (count / len(df_outliers)) * 100
        ax2.text(
            bar.get_x() + bar.get_width()/2, 
            bar.get_height() + 1,
            f'{count} ({pct:.1f}%)', 
            ha='center', 
            va='bottom', 
            fontsize=10,
            fontweight='bold'
        )
    
    # Ajustar layout
    plt.tight_layout()
    
    # Mostrar en Streamlit
    st.pyplot(fig)
    
    # Opcional: Guardar la figura
    # fp_dvd = '/ruta/para/guardar'
    # plt.savefig(fp_dvd + '/figuras/distribucion_grupos_patrones.png', dpi=300, bbox_inches='tight')
    
    
    
    # ============================================
    # PATRONES OUTLIER (MUY RAROS)
    # ============================================
    
    st.subheader("🔍 Patrones Outlier (Muy Raros)")
    
    # Filtrar patrones 'muy raro'
    patrones_muy_raros = df_outliers[df_outliers['grupo_patron'] == 'muy raro']
    
    # Obtener patrones únicos con sus frecuencias
    patrones_unicos = patrones_muy_raros[['patron_ordenado', 'frecuencia_patron', 'frecuencia_pct']].drop_duplicates()
    
    # Ordenar por frecuencia (de mayor a menor)
    patrones_unicos = patrones_unicos.sort_values('frecuencia_pct', ascending=False)
    
    
    
    

    
    
    
    # Mapeo de rareza a número para el eje derecho
    rareza_numerica = {
        'comun': 4,
        'poco comun': 3,
        'raro': 2,
        'muy raro': 1
    }
    
    # ============================================
    # PREPARAR DATOS PARA EL GRÁFICO
    # ============================================
    
    # Obtener todos los patrones con su rareza y frecuencia
    df_patrones_rareza = df_outliers[['patron_ordenado', 'frecuencia_patron', 'frecuencia_pct', 'grupo_patron']].drop_duplicates()
    
    # Ordenar por frecuencia (de mayor a menor)
    df_patrones_rareza = df_patrones_rareza.sort_values('frecuencia_pct', ascending=True)  # True para barras horizontales (arriba los más frecuentes)
    
    # Convertir tuplas a string para mostrar
    df_patrones_rareza['patron_str'] = df_patrones_rareza['patron_ordenado'].apply(
        lambda x: ' - '.join(x) if isinstance(x, tuple) else str(x)
    )
    
    # Asignar color según rareza
    df_patrones_rareza['color'] = df_patrones_rareza['grupo_patron'].map(colors_rareza)
    
    # Asignar valor numérico para el eje derecho
    df_patrones_rareza['rareza_num'] = df_patrones_rareza['grupo_patron'].map(rareza_numerica)
    
    # ============================================
    # CREAR GRÁFICO DE BARRAS HORIZONTALES
    # ============================================
    
    fig, ax1 = plt.subplots(figsize=(12, max(6, len(df_patrones_rareza) * 0.3)))
    
    # Barra horizontal: frecuencia
    bars = ax1.barh(
        df_patrones_rareza['patron_str'],
        df_patrones_rareza['frecuencia_pct'],
        color=df_patrones_rareza['color'],
        edgecolor='black',
        alpha=0.8,
        linewidth=1.5,
        height=0.7
    )
    
    # Configurar eje X (frecuencia)
    ax1.set_xlabel('Frecuencia (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Patrón', fontsize=12, fontweight='bold')
    ax1.set_title('Distribución de Patrones por Frecuencia y Rareza', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Añadir etiquetas de porcentaje al final de cada barra
    for bar, pct in zip(bars, df_patrones_rareza['frecuencia_pct']):
        ax1.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height()/2,
            f'{pct:.1f}%',
            va='center',
            fontsize=9,
            fontweight='bold'
        )
    
    # ============================================
    # EJE Y DERECHO: NIVEL DE RAREZA
    # ============================================
    
    ax2 = ax1.twinx()
    ax2.set_ylim(ax1.get_ylim())
    
    # Crear etiquetas para el eje derecho
    etiquetas_rareza = []
    for rareza in df_patrones_rareza['grupo_patron']:
        # Capitalizar primera letra
        etiquetas_rareza.append(rareza.capitalize())
    
    # Colores para las etiquetas del eje derecho
    colores_etiquetas = [colors_rareza[r] for r in df_patrones_rareza['grupo_patron']]
    
    # Configurar eje derecho (rareza)
    ax2.set_ylabel('Nivel de Rareza', fontsize=12, fontweight='bold', color='black')
    ax2.set_yticks(range(len(df_patrones_rareza)))
    ax2.set_yticklabels(etiquetas_rareza, fontsize=10, fontweight='bold')
    
    # Colorear las etiquetas del eje derecho
    for tick, color in zip(ax2.get_yticklabels(), colores_etiquetas):
        tick.set_color(color)
    
    # Ajustar límites del eje derecho para que coincida con el izquierdo
    ax2.set_ylim(ax1.get_ylim())
    
    # Quitar el grid del eje derecho para no solapar
    ax2.grid(False)
    
    
    
    plt.tight_layout()
    st.pyplot(fig)
    
    
        
    # Leyenda
    st.markdown("**📊 Leyenda de colores por rareza:**")
    cols = st.columns(4)
    for i, (grupo, color) in enumerate(colors_rareza.items()):
        with cols[i]:
            st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="display: inline-block; width: 20px; height: 20px; background-color: {color}; border-radius: 4px; border: 1px solid #ddd;"></span>
                <span style="font-size: 14px;"><b>{grupo}</b></span>
            </div>
            """, unsafe_allow_html=True)    
        
       
            
            
        
        
    
    # ARBOL
    col1 = st.columns(1)[0]
    st.subheader("🌳 Clustering con Árbol de Decisión")
    
    
    def crear_arbol_plotly(tree_data, feature_names, class_names):
        """
        Crea un gráfico interactivo del árbol usando Plotly con layout jerárquico
        """
        children_left = tree_data['children_left']
        children_right = tree_data['children_right']
        feature = tree_data['feature']
        threshold = tree_data['threshold']
        value = tree_data['value']
        n_node_samples = tree_data['n_node_samples']
        
        # Crear grafo
        G = nx.DiGraph()
        
        def add_nodes(node_id, depth=0, parent=None, edge_label=""):
            if children_left[node_id] == children_right[node_id]:
                # Nodo hoja - obtener la clase mayoritaria
                class_counts = value[node_id][0]
                class_idx = np.argmax(class_counts)
                class_name = class_names[class_idx]
                
                # Asignar color según rareza de la clase
                if class_name in colors_rareza:
                    node_color = colors_rareza[class_name]
                else:
                    node_color = '#90EE90'  # Color por defecto (verde claro)
                
                label = f"{class_name}<br>n={n_node_samples[node_id]}"
                G.add_node(node_id, label=label, is_leaf=True, depth=depth, 
                          samples=n_node_samples[node_id], class_name=class_name,
                          color=node_color)
            else:
                # Nodo interno
                label = f"{feature_names[feature[node_id]]}<br>≤ {threshold[node_id]:.2f}"
                G.add_node(node_id, label=label, is_leaf=False, depth=depth,
                          samples=n_node_samples[node_id], color='#ADD8E6')  # Azul claro por defecto
                
                add_nodes(children_left[node_id], depth+1, node_id, f"≤{threshold[node_id]:.2f}")
                add_nodes(children_right[node_id], depth+1, node_id, f">{threshold[node_id]:.2f}")
            
            if parent is not None:
                G.add_edge(parent, node_id, label=edge_label)
        
        add_nodes(0)
        
        # ============================================
        # LAYOUT JERÁRQUICO MANUAL
        # ============================================
        
        # Calcular profundidad máxima
        max_depth = max([G.nodes[n]['depth'] for n in G.nodes()])
        
        # Posiciones manuales
        pos = {}
        
        # Función recursiva para posicionar nodos
        def compute_pos(node_id, x, y, x_range):
            pos[node_id] = (x, y)
            
            # Obtener hijos (excluyendo al padre)
            children = [n for n in G.neighbors(node_id) if G.nodes[n]['depth'] > G.nodes[node_id]['depth']]
            
            if not children:
                return
            
            # Distribuir hijos horizontalmente
            if len(children) == 2:
                child_x_range = x_range / 2
                compute_pos(children[0], x - child_x_range/2, y - 1, child_x_range)
                compute_pos(children[1], x + child_x_range/2, y - 1, child_x_range)
            elif len(children) == 1:
                compute_pos(children[0], x, y - 1, x_range)
        
        # Empezar desde la raíz (nodo 0)
        compute_pos(0, 0, 0, 4)
        
        # Normalizar posiciones para que estén centradas
        all_x = [pos[n][0] for n in pos]
        all_y = [pos[n][1] for n in pos]
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        
        # Escalar para mejor visualización
        for node in pos:
            x_norm = (pos[node][0] - min_x) / (max_x - min_x) * 2 - 1 if max_x != min_x else 0
            y_norm = (pos[node][1] - min_y) / (max_y - min_y) * 2 - 1 if max_y != min_y else 0
            pos[node] = (x_norm * 5, y_norm * 3)  # Escalar para mejor visualización
        
        # Crear figura Plotly
        fig = go.Figure()
        
        # Dibujar aristas (conexiones entre nodos)
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            fig.add_trace(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=1.5, color='gray'),
                hoverinfo='none'
            ))
        
        # Dibujar nodos
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        node_sizes = []
        node_ids = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(G.nodes[node]['label'])
            node_colors.append(G.nodes[node]['color'])
            node_sizes.append(40 + G.nodes[node]['samples'] / 15)
            node_ids.append(node)
        
        fig.add_trace(go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(color='black', width=1.5)
            ),
            text=node_text,
            textposition='middle center',
            textfont=dict(size=9, color='black'),
            hoverinfo='text',
            hovertext=[f"Nodo: {node_ids[i]}<br>Muestras: {G.nodes[node]['samples']}<br>Profundidad: {G.nodes[node]['depth']}<br>Clase: {G.nodes[node].get('class_name', 'Interno')}" 
                       for i, node in enumerate(G.nodes())]
        ))
        
        fig.update_layout(
            title='Árbol de Decisión - Predicción de Grupo de Patrón',
            showlegend=False,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-6, 6]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-5, 1.5]),
            height=800,
            width=1100,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        return fig
    
    # Crear y mostrar el árbol
    fig = crear_arbol_plotly(
        results['tree'],
        results['feature_names'],
        results['class_names']
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    
    # Después de st.plotly_chart, añadir leyenda
    st.markdown("**📊 Leyenda de colores por rareza:**")
    cols = st.columns(4)
    for i, (grupo, color) in enumerate(colors_rareza.items()):
        with cols[i]:
            st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="display: inline-block; width: 20px; height: 20px; background-color: {color}; border-radius: 4px; border: 1px solid #ddd;"></span>
                <span style="font-size: 14px;"><b>{grupo}</b></span>
            </div>
            """, unsafe_allow_html=True)
    
# TAB 5: Predicciones
with tab5:
    
    
    st.title("Predicciones PREx vs Diagnósticos de D1")
    
    
    

    
    col1, col2 = st.columns(2)
    
    with col1:
        modelo_seleccionado = st.selectbox(
            "Selecciona modelo:",
            results['pre_cols']
        )
    
    with col2:
        grupo_seleccionado = st.selectbox(
            "Selecciona grupo de patrón:",
            ['Todos'] + results['grupos']
        )
    
    # ============================================
    # FILTRAR DATOS
    # ============================================
    
    df_plot = results['scatter_data'][modelo_seleccionado].copy()
    
    if grupo_seleccionado != 'Todos':
        df_plot = df_plot[df_plot['grupo_patron'] == grupo_seleccionado]
    
    # ============================================
    # SCATTERPLOT INTERACTIVO
    # ============================================
    
    st.subheader(f"Instancias correctos/incorrectos: {modelo_seleccionado} vs D1")
    
    # Crear hover personalizado
    df_plot['hover_text'] = df_plot.apply(
        lambda row: f"ID: {row['ID']}<br>D1: {row['D1']}<br>{modelo_seleccionado}: {row[modelo_seleccionado]}<br>Grupo: {row['grupo_patron']}<br>Aciertos: {row['num_aciertos']}/5<br>Tasa: {row['tasa_promedio']:.1f}%",
        axis=1
    )
    
    fig = px.scatter(
        df_plot,
        x='paciente',
        y=modelo_seleccionado,
        color='color',
        color_discrete_map={'Correcto': 'green', 'Incorrecto': 'red'},
        hover_data={
            'ID': True,
            'D1': True,
            modelo_seleccionado: True,
            'grupo_patron': True,
            'num_aciertos': True,
            'tasa_promedio': ':.1f',
            'paciente': False,
            'color': False
        },
        title=f'{modelo_seleccionado} vs D1 - Grupo: {grupo_seleccionado}',
        labels={
            'paciente': 'Paciente',
            modelo_seleccionado: modelo_seleccionado,
            'color': 'Estado'
        }
    )
    
    fig.update_traces(
        marker=dict(size=14, line=dict(width=1, color='black')),
        selector=dict(mode='markers')
    )
    
    fig.update_layout(
        height=600,
        hovermode='closest',
        legend=dict(
            yanchor="top",
            y=1.3,
            xanchor="center",
            x=0.5
        ),
        # === HABILITAR ZOOM ===
        dragmode='zoom',           # Modo de arrastre: zoom
        xaxis=dict(
            fixedrange=False,      # Permitir zoom en X
            constrain='domain'     # Mantener proporción
        ),
        yaxis=dict(
            fixedrange=False,      # Permitir zoom en Y
            constrain='domain'
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.divider()
    
    # Comparar los cincos PREx mediante 5 graficos
    
    st.subheader("Aciertos/Errores por Modelo y Diagnóstico")
    
    # Crear 5 columnas (una por cada PRE)
    cols = st.columns(5)
    
    for idx, pre in enumerate(['PRE1', 'PRE2', 'PRE3', 'PRE4', 'PRE5']):
        with cols[idx]:
            st.markdown(f"**{pre}**")
            
            # Preparar datos para este modelo
            df_temp = results['df_outliers'].copy()
            df_temp['acierto'] = df_temp[pre] == df_temp['D1']
            
            # Agrupar por D1 y acierto
            grouped = df_temp.groupby(['D1', 'acierto']).size().unstack(fill_value=0)
            
            # Asegurar que existen todas las clases
            for dx in ['ED1', 'ED2', 'ED3']:
                if dx not in grouped.index:
                    grouped.loc[dx] = [0, 0]
            
            # Reordenar
            grouped = grouped.reindex(['ED1', 'ED2', 'ED3'])
            
            # Separar correctos e incorrectos
            correctos = grouped.get(True, pd.Series(0, index=grouped.index))
            incorrectos = grouped.get(False, pd.Series(0, index=grouped.index))
            
            # Crear gráfico de barras apiladas
            fig, ax = plt.subplots(figsize=(4, 6))
            
            # Barra inferior: Incorrectos (rojo)
            ax.bar(grouped.index, incorrectos, 
                   color='red', label='Incorrectos', alpha=0.8, edgecolor='black')
            
            # Barra superior: Correctos (verde), apilada sobre incorrectos
            ax.bar(grouped.index, correctos, 
                   bottom=incorrectos, color='green', label='Correctos', 
                   alpha=0.8, edgecolor='black')
            
            # Configurar gráfico
            ax.set_xlabel('Clase', fontsize=9)
            ax.set_ylabel('Pacientes', fontsize=9)
            ax.set_title(f'{pre}', fontsize=10, fontweight='bold')
            ax.legend(loc='upper right', fontsize=7)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Añadir valores en las barras
            for i, (dx, corr, incorr) in enumerate(zip(grouped.index, correctos, incorrectos)):
                total = corr + incorr
                if total > 0:
                    # Texto del total encima de la barra
                    ax.text(i, total + 0.5, str(total), ha='center', va='bottom', fontsize=8, fontweight='bold')
                    # Texto de correctos dentro de la barra verde
                    if corr > 0:
                        ax.text(i, incorr + corr/2, f'{corr}', ha='center', va='center', fontsize=7, color='white', fontweight='bold')
                    # Texto de incorrectos dentro de la barra roja
                    if incorr > 0:
                        ax.text(i, incorr/2, f'{incorr}', ha='center', va='center', fontsize=7, color='white', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    
    st.divider()
    
    
    
   
    
   
    # Comparar tasa de aciertos de 5 modelos :grafico de 5 columnas
   
   
    st.subheader("📊 Comparación de todos los modelos")
    
    if grupo_seleccionado == 'Todos':
        df_temp = results['df_outliers']
    else:
        df_temp = results['df_outliers'][results['df_outliers']['grupo_patron'] == grupo_seleccionado]
    
    tasa_modelos = {}
    for pre in results['pre_cols']:
        aciertos = (df_temp[pre] == df_temp['D1']).sum()
        tasa_modelos[pre] = aciertos / len(df_temp) * 100
    
    df_barras = pd.DataFrame({
        'Modelo': list(tasa_modelos.keys()),
        'Tasa_Acierto': list(tasa_modelos.values())
    })
    
    fig2 = px.bar(
        df_barras,
        x='Modelo',
        y='Tasa_Acierto',
        color='Tasa_Acierto',
        color_continuous_scale='RdYlGn',
        title=f'Tasa de Acierto por Modelo - Grupo: {grupo_seleccionado}',
        labels={'Tasa_Acierto': 'Tasa de Acierto (%)'},
        text='Tasa_Acierto'
    )
    
    fig2.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig2.update_layout(height=400, yaxis_range=[0, 100])
    st.plotly_chart(fig2, use_container_width=True)
        
    st.divider()

   
    # ============================================
    # TABLA RESUMEN CON COLORES (más visual)
    # ============================================
  
    st.subheader("📋 Resumen de Aciertos por Modelo")
    
    tabla_resumen = []
    for pre in results['pre_cols']:
        aciertos = (df_temp[pre] == df_temp['D1']).sum()
        errores = len(df_temp) - aciertos
        tasa = aciertos / len(df_temp) * 100
        
        tabla_resumen.append({
            'Modelo': pre,
            'Aciertos': aciertos,
            'Errores': errores,
            'Tasa Acierto (%)': round(tasa, 1)
        })
    
    df_tabla = pd.DataFrame(tabla_resumen)
    
    

    # Crear tabla HTML manualmente
    # Definir colores
    
    # Crear tabla HTML
    html_tabla = '<table class="tabla-resumen"><tr><th>Modelo</th><th>✅ Aciertos</th><th>❌ Errores</th><th>🎯 Tasa Acierto (%)</th></tr>'
    
    for idx, (_, row) in enumerate(df_tabla.iterrows()):
        # Determinar color para la tasa de acierto
        color_tasa = colors_si_no[1] if row['Tasa Acierto (%)'] >= 60 else colors_si_no[0]
        # Color de fondo alterno
        bg_color = colores_alternas[idx % 2]
        
        html_tabla += f'''
        <tr style="background-color: {bg_color};">
            <td><b>{row["Modelo"]}</b></td>
            <td>{row["Aciertos"]}</td>
            <td>{row["Errores"]}</td>
            <td style="color: {color_tasa}; font-weight: bold;">{row["Tasa Acierto (%)"]}%</td>
        </tr>'''
    
    html_tabla += '</table>'
    
    st.markdown(html_tabla, unsafe_allow_html=True)
    st.caption(f"Total de pacientes en este grupo: {len(df_temp)}")
    
# CASOS DIFICILES
with tab6:
    #col1 = st.columns(1)[0]
    
    
    st.title("🔴 Análisis de Casos Difíciles")
    st.markdown("Identificación de pacientes que son difíciles de clasificar para los modelos")
    
    # ============================================
    # MÉTRICAS PRINCIPALES
    # ============================================
    
    st.subheader("📊 Resumen")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Pacientes", results['total_pacientes'])
    with col2:
        st.metric("🔴 Casos Difíciles", 
                  f"{results['total_dificiles']} ({results['pct_dificiles']:.1f}%)")
    with col3:
        st.metric("🔴🔴 Muy Difíciles", 
                  f"{results['total_muy_dificiles']} ({results['pct_muy_dificiles']:.1f}%)")
    with col4:
        st.metric("🔴🔴🔴 Extremos", 
                  f"{results['total_extremos']} ({results['pct_extremos']:.1f}%)")
    
    st.divider()
    
    # ============================================
    # GRÁFICO 1: DISTRIBUCIÓN DE NIVELES DE DIFICULTAD
    # ============================================
    
    st.subheader("📊 Distribución de Niveles de Dificultad")
    
    # Crear DataFrame para el gráfico
    df_niveles = pd.DataFrame({
        'Nivel': list(results['distribucion_dificultad'].keys()),
        'Pacientes': list(results['distribucion_dificultad'].values()),
        
    })
    
    # Ordenar según niveles_order
    df_niveles['Nivel'] = pd.Categorical(df_niveles['Nivel'], 
                                          categories=results['niveles_order'], 
                                          ordered=True)
    df_niveles = df_niveles.sort_values('Nivel')
    
    
    
    fig1 = px.bar(
        df_niveles,
        x='Nivel',
        y='Pacientes',
        color='Nivel',
        color_discrete_map=colors_niveles,
        title='Distribución de Pacientes por Nivel de Dificultad',
        labels={'Pacientes': 'Número de Pacientes'},
        text='Pacientes'
    )
    #fig1.update_traces(texttemplate='%{text} (%{customdata:.1f}%)',
    fig1.update_traces(texttemplate='%{text} ', 
                       #customdata=df_niveles['Porcentaje'].values,
                       textposition='outside')
    fig1.update_layout(height=450, showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)
    
    # ============================================
    # GRÁFICO 2: CASOS DIFÍCILES POR GRUPO
    # ============================================
    
    st.subheader("📊 Casos Difíciles por Grupo de Patrón")
    
    df_pct_grupo = pd.DataFrame({
        'Grupo': list(results['pct_dificil_grupo'].keys()),
        'Porcentaje': list(results['pct_dificil_grupo'].values())
    })
    
    # Ordenar grupos
    orden_grupos = ['comun', 'poco comun', 'raro', 'muy raro']
    df_pct_grupo['Grupo'] = pd.Categorical(df_pct_grupo['Grupo'], categories=orden_grupos, ordered=True)
    df_pct_grupo = df_pct_grupo.sort_values('Grupo')
    
    #colores_grupos = {'comun': 'green', 'poco comun': 'yellowgreen', 'raro': 'orange', 'muy raro': 'red'}
    
    fig2 = px.bar(
        df_pct_grupo,
        x='Grupo',
        y='Porcentaje',
        color='Grupo',
        color_discrete_map=colors_grupos,
        title='Porcentaje de Casos Difíciles por Grupo de Patrón',
        labels={'Porcentaje': '% de Casos Difíciles'},
        text='Porcentaje'
    )
    
    fig2.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig2.update_layout(height=400, yaxis_range=[0, 100], showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)
    
    # ============================================
    # GRÁFICO 3: CASOS DIFÍCILES POR CLASE
    # ============================================
    
    st.subheader("📊 Casos Difíciles por Clase D1")
    
    df_pct_clase = pd.DataFrame({
        'Clase': list(results['pct_dificil_clase'].keys()),
        'Porcentaje': list(results['pct_dificil_clase'].values())
    })
    
    colores_clases = {'ED1': '#1f77b4', 'ED2': '#ff7f0e', 'ED3': '#2ca02c'}
    
    fig3 = px.bar(
        df_pct_clase,
        x='Clase',
        y='Porcentaje',
        color='Clase',
        color_discrete_map=colors_diag,
        title='Porcentaje de Casos Difíciles por Clase D1',
        labels={'Porcentaje': '% de Casos Difíciles'},
        text='Porcentaje'
    )
    
    fig3.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig3.update_layout(height=400, yaxis_range=[0, 100], showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)
    
    # ============================================
    # TABLA DE CASOS DIFÍCILES
    # ============================================
    
    st.divider()
    st.subheader("📋 Listado de Casos Difíciles de D1")
    
    # Mostrar opciones de filtro
    ver_tipo = st.radio(
        "Selecciona el tipo de casos a mostrar:",
        ['Todos los casos difíciles', 'Muy difíciles (4+ errores)', 'Extremos (5 errores)']
    )
    
    if ver_tipo == 'Todos los casos difíciles':
        df_mostrar = results['casos_dificiles']
    elif ver_tipo == 'Muy difíciles (4+ errores)':
        df_mostrar = results['casos_muy_dificiles']
    else:
        df_mostrar = results['casos_extremos']
    
    if len(df_mostrar) > 0:
        df_display = df_mostrar[['ID', 'D1', 'PRE1', 'PRE2', 'PRE3', 'PRE4', 'PRE5', 
                                'num_aciertos', 'num_errores', 'grupo_patron', 'nivel_dificultad']]
        
        # Colorear según nivel de dificultad
        def color_por_nivel(row):
            nivel = row['nivel_dificultad']
            if nivel == 'Extremadamente Difícil':
                return ['background-color: #ffcccc; color: #8B0000; font-weight: bold'] * len(row)
            elif nivel == 'Muy Difícil':
                return ['background-color: #ffe6cc; color: #CC5500; font-weight: bold'] * len(row)
            elif nivel == 'Difícil':
                return ['background-color: #fff3cc; color: #8B7500'] * len(row)
            return [''] * len(row)
        
        styled_df = df_display.style.apply(color_por_nivel, axis=1)
        
        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                'ID': 'ID',
                'D1': 'D1',
                'PRE1': 'PRE1',
                'PRE2': 'PRE2',
                'PRE3': 'PRE3',
                'PRE4': 'PRE4',
                'PRE5': 'PRE5',
                'num_aciertos': 'Aciertos',
                'num_errores': 'Errores',
                'grupo_patron': 'Grupo Patrón',
                'nivel_dificultad': 'Nivel'
            }
        )
        st.caption(f"Mostrando {len(df_mostrar)} pacientes")

       # Leyenda
        st.markdown("**📊 Leyenda de colores por nivel de dificultad:**")
        cols_legend = st.columns(3)
        with cols_legend[0]:
            st.markdown("""
            <div style="display: flex; align-items: center; gap: 8px; margin: 5px 0;">
                <span style="display: inline-block; width: 20px; height: 20px; background-color: #fff3cc; border-radius: 4px; border: 1px solid #ddd;"></span>
                <span style="font-size: 14px;">Difícil</span>
            </div>
            """, unsafe_allow_html=True)
        with cols_legend[1]:
            st.markdown("""
            <div style="display: flex; align-items: center; gap: 8px; margin: 5px 0;">
                <span style="display: inline-block; width: 20px; height: 20px; background-color: #ffe6cc; border-radius: 4px; border: 1px solid #ddd;"></span>
                <span style="font-size: 14px;">Muy Difícil</span>
            </div>
            """, unsafe_allow_html=True)
        with cols_legend[2]:
            st.markdown("""
            <div style="display: flex; align-items: center; gap: 8px; margin: 5px 0;">
                <span style="display: inline-block; width: 20px; height: 20px; background-color: #ffcccc; border-radius: 4px; border: 1px solid #ddd;"></span>
                <span style="font-size: 14px;">Extremadamente Difícil</span>
            </div>
            """, unsafe_allow_html=True)
               
        
       

                
                
                
            
with st.sidebar:
    
    st.subheader(" Vista Raw")
    st.markdown(f"""
    - **Pacientes:** {results['dimensiones'][0]}
    - **Features:** {results['dimensiones'][1]}
    - **Diagnósticos:** {results['unicos_diag']}
    - **Médicos:** {results['medicos']}
    - **Predicciones:** {results['predicciones']}
    """)
    
    st.divider()
    
   
    st.subheader("Limpieza de datos")
    st.markdown(f""" 
        - **Features:**
        - **ID**
        - **Edad**
        - **Sexo**
        - **D1..D5: Medicos emitiendo un Diagnóstico**
        - **PRE1..PRE5: Modelos prediciendo un Diagnóstico**
        - **P1..P7:Puntuación en distintas pruebas**
        - **D1.1-D5.1:Duración total de una prueba**       
        - **S1-S9:Suma de tiempos de respuesta parciales**              
        - **MNx:Mínimo de tiempos de respuesta parciales**
        - **MXx:Máximo de tiempos de respuesta parciales**
        - **MDx:Media de tiempos de respuesta parciales**
    """)
    
    st.divider()
    
    st.subheader(" Edad-Sexo")
    st.markdown(f"""
        - **Valores Únicos Sexo:**{results['unicos_sexo']}
        - **Valores Ünicos Edad:**{results['unicos_edad']}                
    """)            
    st.divider()

    st.subheader(" Diagnósticos")
    
   
    
    st.markdown(f"""
        - **Médicos:**{results['medicos']}
        - **Diagnósticos Únicos: ED1, ED2, ED3**
        - **Patrones de Diagnósticos:6**
    """)             
    st.divider()

    st.subheader("Predicciones")
     
    st.divider()

    st.subheader(" Casos Difíciles")
    
    st.markdown("### 📌 Niveles de Dificultad")
    st.markdown(f"""
        - **Facil:3 o más aciertos**       
        - **🔴Difícil:2 aciertos**
        - **🔴🔴Muy Difícil:1 acierto**
        - **🔴🔴🔴 Extremadamente Difícil:0 aciertos**
    """)             
    st.divider()
  
    

    
    





# ============================================
# PIE DE PÁGINA
# ============================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; font-size: 12px;">
    📊 Dashboard de Visualización de Datos - Práctica Final | Creado con Streamlit
</div>
""", unsafe_allow_html=True)