import gradio as gr
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from src.chatbot.engine import responder_pregunta
from src.analysis.visualizer import (
    obtener_graficos_categoricos,
    obtener_graficos_numericos,
    obtener_graficos_analisis_target,
    obtener_graficos_correlacion,
    obtener_grafico_nulos
)

def load_resources():
    modelo = joblib.load('models/modelo_solar.pkl')
    prepro = joblib.load('models/preprocesador.pkl')
    df = pd.read_excel('data/energia_solar_pereira_colombia_clean.xlsx')
    return modelo, prepro, df

def predecir_ahorro(anio, tipo, material, paneles, radiacion, eficiencia, humedad, temperatura):
    modelo, prepro, df = load_resources()
    
    # Cargar indicadores economicos para el año seleccionado
    ruta_ind = os.path.join("data", "colombia_indicadores_2018_2100.xlsx")
    df_ind = pd.read_excel(ruta_ind)
    
    # Buscar datos del año (si no existe, usamos el ultimo conocido)
    datos_anio = df_ind[df_ind.iloc[:, 0] == anio]
    if datos_anio.empty:
        datos_anio = df_ind.tail(1)
        
    ipc_g = datos_anio['ipc_general_pct'].values[0]
    ipc_e = datos_anio['ipc_energia_pct'].values[0]
    trm = datos_anio['trm_promedio_cop'].values[0]

    # Crear DataFrame con todas las columnas que el modelo espera
    input_df = pd.DataFrame([[
        anio, tipo, material, paneles, radiacion, 
        eficiencia, humedad, temperatura,
        ipc_g, ipc_e, trm
    ]], columns=[
        'Año Instalación', 'Tipo', 'Material Panel', 'N° Paneles', 'Radiación Solar',
        'Eficiencia Panel (%)', 'Humedad Relativa Prom', 'Temperatura Prom',
        'ipc_general_pct', 'ipc_energia_pct', 'trm_promedio_cop'
    ])
    
    input_pre = prepro.transform(input_df)
    valor_predicho = modelo.predict(input_pre)[0]

    # Formatear el resultado con lenguaje amigable y llamativo
    explicacion = f"## 💰 Tu ahorro estimado: **${valor_predicho:,.2f} COP** mensuales\n"
    explicacion += f"Este calculo considera la **inflacion de energia ({ipc_e}%)** y la **TRM (${trm:,.0f})** para el año {anio}.\n\n"
    
    # Añadir un indicador visual de calidad
    if valor_predicho > 500000:
        explicacion += "🌟 ¡Excelente! Este es un ahorro muy significativo."
    elif valor_predicho > 100000:
        explicacion += "✅ ¡Muy bien! Estas teniendo un ahorro considerable."
    else:
        explicacion += "💡 Es un ahorro inicial, ¡cada peso cuenta!"

    if anio > 2025:
        explicacion += f"\n\n*Nota: Proyeccion basada en los ultimos indicadores economicos disponibles.*"

    # Creamos un Medidor (Gauge) visual
    fig, ax = plt.subplots(figsize=(6, 2))
    max_ahorro_esperado = 1000000 # Un millon como tope para la escala
    porcentaje = min((valor_predicho / max_ahorro_esperado) * 100, 100)
    
    # Dibujar barra de fondo y barra de progreso
    ax.barh([0], [100], color='#eeeeee', height=0.4)
    color_bar = '#4CAF50' if porcentaje > 60 else '#FFC107' if porcentaje > 30 else '#F44336'
    ax.barh([0], [porcentaje], color=color_bar, height=0.4)
    
    ax.set_xlim(0, 100)
    ax.set_axis_off()
    ax.text(0, 0.35, "Nivel de Ahorro", fontsize=12, fontweight='bold')
    ax.text(porcentaje, -0.35, f"{porcentaje:.1f}%", ha='center', fontweight='bold', color=color_bar)
    
    plt.tight_layout()
    return explicacion, fig

def build_app():
    plt.close('all') # Limpiar figuras previas para evitar warnings de memoria
    modelo, prepro, df = load_resources()

    with gr.Blocks() as demo:
        gr.Markdown("# ☀️ Calculadora de Energia Solar Inteligente (Pereira)")

        with gr.Tabs():
            with gr.TabItem("💬 Preguntale a la IA"):
                gr.ChatInterface(fn=lambda m, h: responder_pregunta(m, h, df))

            with gr.TabItem("📊 Simular mi Ahorro"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🛠️ Datos de tu Instalacion")
                        anio = gr.Number(label="¿En que año proyectamos?", value=2024)
                        tipo = gr.Dropdown(choices=list(df['Tipo'].unique()), label="Tipo de instalacion")
                        material = gr.Dropdown(choices=list(df['Material Panel'].unique()), label="Material de los paneles")
                        paneles = gr.Slider(1, 100, label="Cantidad de paneles", value=10)
                        radiacion = gr.Slider(1, 10, label="Nivel de sol (Radiacion)", value=5.5)
                        
                        with gr.Accordion("⚙️ Datos tecnicos avanzados", open=False):
                            eficiencia = gr.Slider(10, 25, label="Eficiencia del panel (%)", value=18.5)
                            humedad = gr.Slider(30, 90, label="Humedad relativa (%)", value=75.0)
                            temperatura = gr.Slider(15, 35, label="Temperatura promedio (°C)", value=22.0)
                        
                        btn = gr.Button("¡Predecir mi ahorro!", variant="primary")
                    with gr.Column():
                        gr.Markdown("### 📈 Resultado de tu ahorro")
                        res = gr.Markdown("Aqui aparecera tu resultado...")
                        plt_plot = gr.Plot()
                
                btn.click(
                    predecir_ahorro, 
                    inputs=[anio, tipo, material, paneles, radiacion, eficiencia, humedad, temperatura], 
                    outputs=[res, plt_plot]
                )

            with gr.TabItem("📊 Dashboard de Analisis"):
                gr.Markdown("## 📈 Analisis de la Base de Datos Solar")
                
                with gr.Tabs():
                    with gr.Tab("General"):
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("### Matriz de Correlacion")
                                gr.Plot(value=obtener_graficos_correlacion(df))
                            with gr.Column():
                                gr.Markdown("### Calidad de Datos (Nulos)")
                                gr.Plot(value=obtener_grafico_nulos(df))
                    
                    with gr.Tab("Distribuciones"):
                        gr.Markdown("### Variables Categoricas Principales")
                        cat_figs = obtener_graficos_categoricos(df)
                        with gr.Row():
                            if len(cat_figs) > 0: gr.Plot(value=cat_figs[0], label="Año Instalacion")
                            if len(cat_figs) > 2: gr.Plot(value=cat_figs[2], label="Material Panel")
                            if len(cat_figs) > 3: gr.Plot(value=cat_figs[3], label="Tipo de Instalacion")
                        
                        gr.Markdown("### Variables Numericas (Distribucion)")
                        num_figs = obtener_graficos_numericos(df)
                        with gr.Row():
                            # Mostrar 2 graficos numericos interesantes (ej. Radiacion y Humedad)
                            # Asumiendo que existen por el dataset
                            if len(num_figs) > 4: gr.Plot(value=num_figs[4], label="Radiacion Solar")
                            if len(num_figs) > 6: gr.Plot(value=num_figs[6], label="Humedad Relativa")

                    with gr.Tab("Analisis de Ahorro"):
                        gr.Markdown("### Comportamiento del Ahorro Mensual")
                        target_figs = obtener_graficos_analisis_target(df)
                        with gr.Row():
                            for fig in target_figs:
                                gr.Plot(value=fig)

            with gr.TabItem("📘 Acerca de"):
                gr.Markdown("### ☀️ Sobre este Proyecto")
                gr.Markdown("""
                Esta aplicacion utiliza Inteligencia Artificial para predecir el ahorro economico de instalaciones solares en Pereira, Colombia.
                
                **Caracteristicas clave:**
                - **Chatbot Inteligente:** Resuelve tus dudas sobre energia solar.
                - **Simulador Avanzado:** Predicciones ajustadas a la inflacion (IPC) y TRM.
                - **Dashboard Integrado:** Visualizacion completa de los datos recolectados.
                
                *Desarrollado para el Diplomado en IA - TalentoTech.*
                """)

    return demo

