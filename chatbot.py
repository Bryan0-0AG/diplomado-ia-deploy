from src.interfaces.gradio_app import build_app
import gradio as gr

if __name__ == "__main__":
    app = build_app()
    # CORRECCION GRADIO 6: El tema se pasa aqui
    # He puesto share=False por defecto por el tema del antivirus
    app.launch(share=False, theme=gr.themes.Soft())
