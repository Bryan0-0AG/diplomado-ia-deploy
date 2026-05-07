import gradio as gr
import os
from modules.chatbot_logic import responder_pregunta
from modules.data_cleaning import limpiar_datos

# Cargamos datos una vez al iniciar para que el bot tenga contexto
ruta_data = 'assets/energia_solar_pereira_colombia.xlsx'
if os.path.exists(ruta_data):
    df_context = limpiar_datos(ruta_data)
else:
    df_context = None

def predict(message, history):
    # Pasamos el mensaje, el historial y el dataframe cargado
    return responder_pregunta(message, history, df_context)

# Interfaz de Gradio
demo = gr.ChatInterface(
    fn=predict, 
    title="Chatbot Solar ☀️",
    description="Preguntame sobre el analisis de energia solar en Pereira.",
    examples=["¿Cuantos registros hay?", "¿Cual es el ahorro promedio?", "¿Quien es la empresa principal?"]
)

if __name__ == "__main__":
    demo.launch()
