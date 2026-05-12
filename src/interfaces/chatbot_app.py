import gradio as gr
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.chatbot.engine import responder_pregunta
from src.processing.cleaner import limpiar_datos

ruta_data = os.path.join("data", "energia_solar_pereira_colombia.xlsx")
df = limpiar_datos(ruta_data) if os.path.exists(ruta_data) else None

def predict(message, history):
    return responder_pregunta(message, history, df)

def run():
    demo = gr.ChatInterface(fn=predict, title="Chat Solar Modular ☀️")
    demo.launch()

if __name__ == "__main__":
    run()
