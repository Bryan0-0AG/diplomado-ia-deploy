import pandas as pd
import os

def limpiar_datos(ruta_archivo):
    # Carga de datos
    # Ajustamos para que funcione localmente si es necesario, 
    # o mantenemos la logica de carga
    try:
        full_df = pd.read_excel(ruta_archivo)
    except Exception as e:
        print(f"Error al cargar el archivo: {e}")
        return None

    # Filtrado inicial segun el script original
    full_df = full_df[full_df['Nombre Instalación'] != 'Granja Solar Energía de Pereira']
    df = full_df.copy()

    # Reporte basico
    duplicados = df.duplicated().sum()
    nulos = df.isnull().sum()
    print(f"\n{'-'*100}")
    print(f"Duplicados: {duplicados}\n--- NULOS---\n{nulos}")
    print(f"{'-'*100}\n")

    # Limpieza: Convertir strings a minusculas
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.lower()

    # Guardar copia limpia
    nombre_base = os.path.splitext(ruta_archivo)[0]
    ruta_limpia = f"{nombre_base}_clean.xlsx"
    df.to_excel(ruta_limpia, index=False)
    print(f"Archivo limpio guardado en: {ruta_limpia}")

    return df
