import streamlit as st
import pandas as pd
import os
from modules.data_cleaning import limpiar_datos
from modules.visualizations import (
    obtener_graficos_categoricos,
    obtener_graficos_numericos,
    obtener_graficos_analisis_target,
    obtener_graficos_correlacion,
    obtener_grafico_nulos
)

def main():
    st.set_page_config(page_title="Dashboard Energia Solar", layout="wide")
    
    st.title("☀️ Dashboard de Analisis - Energia Solar Pereira")
    st.markdown("---")

    # Sidebar para configuracion
    st.sidebar.header("Configuracion")
    ruta_data = st.sidebar.text_input("Ruta del archivo Excel", "assets/energia_solar_pereira_colombia.xlsx")

    if not os.path.exists(ruta_data):
        st.error(f"No se encontro el archivo en: {ruta_data}")
        return

    # Procesamiento de datos
    if st.sidebar.button("Procesar Datos"):
        with st.spinner("Limpiando datos..."):
            df_clean = limpiar_datos(ruta_data)
            st.session_state['df_clean'] = df_clean
            st.success("Datos procesados correctamente!")

    if 'df_clean' in st.session_state:
        df = st.session_state['df_clean']

        # Tabs para organizar el contenido
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Distribuciones", "📈 Analisis Numerico", "🎯 Variable Objetivo", "🧹 Calidad de Datos"])

        with tab1:
            st.header("Distribuciones Categoricas")
            figs = obtener_graficos_categoricos(df)
            cols = st.columns(2)
            for i, fig in enumerate(figs):
                cols[i % 2].pyplot(fig)

        with tab2:
            st.header("Analisis de Variables Numericas")
            figs = obtener_graficos_numericos(df)
            for fig in figs:
                st.pyplot(fig)
            
            st.header("Matriz de Correlacion")
            st.pyplot(obtener_graficos_correlacion(df))

        with tab3:
            st.header("Analisis de Factura Ahorrada Mensual")
            figs = obtener_graficos_analisis_target(df)
            for fig in figs:
                st.pyplot(fig)

        with tab4:
            st.header("Estado de los Datos")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Mapa de Calor de Nulos")
                st.pyplot(obtener_grafico_nulos(df))
            
            with col2:
                st.subheader("Resumen Estadistico")
                st.write(df.describe())

if __name__ == "__main__":
    main()
