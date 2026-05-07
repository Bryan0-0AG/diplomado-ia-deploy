import pandas as pd

def responder_pregunta(mensaje, historial, df):
    mensaje = mensaje.lower()
    
    if df is None:
        return "Lo siento, aun no he cargado los datos para responderte."

    # Respuestas simples basadas en los datos
    if "registros" in mensaje or "datos" in mensaje or "cuantos" in mensaje:
        return f"El dataset contiene {len(df)} registros de instalaciones solares."
    
    if "empresa" in mensaje or "instala" in mensaje:
        top_empresa = df['Empresa Instaladora'].value_counts().idxmax()
        return f"La empresa que mas ha instalado es '{top_empresa}'."
    
    if "ahorro" in mensaje or "dinero" in mensaje or "factura" in mensaje:
        promedio = df['Factura Ahorrada Mensual'].mean()
        return f"El ahorro promedio registrado es de ${promedio:,.2f} mensuales."
    
    if "hola" in mensaje or "buenos" in mensaje:
        return "¡Hola! Soy tu asistente solar. Puedo darte estadisticas basicas sobre el proyecto. ¿En que puedo ayudarte?"

    return "Por ahora mis conocimientos son limitados a estadisticas basicas de ahorro, empresas y cantidad de registros. ¡Pronto aprendere mas!"
