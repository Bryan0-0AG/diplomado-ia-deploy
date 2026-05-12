import streamlit as st
import pandas as pd
import os
import sys

# Añadir la carpeta src al path para que funcionen los imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.processing.cleaner import limpiar_datos
from src.analysis.visualizer import (
    obtener_graficos_categoricos,
    obtener_graficos_numericos,
    obtener_graficos_analisis_target,
    obtener_graficos_correlacion,
    obtener_grafico_nulos
)

def run():
    st.set_page_config(page_title="Dashboard Solar", layout="wide")
    st.title("☀️ Dashboard Modular de Energia Solar")
    
    ruta_data = os.path.join("data", "energia_solar_pereira_colombia.xlsx")
    ruta_clean = os.path.join("data", "energia_solar_pereira_colombia_clean.xlsx")

    # Carga automatica si ya existe el archivo limpio
    if 'df' not in st.session_state and os.path.exists(ruta_clean):
        st.session_state['df'] = pd.read_excel(ruta_clean)

    if not os.path.exists(ruta_data):
        st.error(f"No se encontró el archivo original en: {ruta_data}")
        if 'df' not in st.session_state:
            return

    with st.sidebar:
        st.header("Configuracion")
        if st.button("🔄 Forzar Reprocesamiento"):
            df = limpiar_datos(ruta_data)
            st.session_state['df'] = df
            st.success("Datos procesados correctamente!")

    if 'df' in st.session_state:
        df = st.session_state['df']
        tab1, tab2 = st.tabs(["Distribuciones", "Analisis"])
        
        with tab1:
            figs = obtener_graficos_categoricos(df)
            for fig in figs: st.pyplot(fig)
        
        with tab2:
            st.pyplot(obtener_graficos_correlacion(df))

if __name__ == "__main__":
    run()
