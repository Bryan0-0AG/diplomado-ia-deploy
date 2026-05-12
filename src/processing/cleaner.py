import pandas as pd
import numpy as np
import os

def limpiar_datos(ruta_solar):
    ruta_indicadores = os.path.join("data", "colombia_indicadores_2018_2100.xlsx")
    
    try:
        df_solar = pd.read_excel(ruta_solar)
        df_indicadores = pd.read_excel(ruta_indicadores)
    except Exception as e:
        print(f"Error al cargar archivos: {e}")
        return None

    # Limpieza inicial solar
    df_solar = df_solar[df_solar['Nombre Instalación'] != 'Granja Solar Energía de Pereira']
    
    # Estandarizar nombres de columnas para el merge
    # Nota: El archivo de indicadores tiene 'ao' (posible encoding)
    # Lo renombramos para estar seguros
    df_indicadores.columns = ['anio_merge'] + list(df_indicadores.columns[1:])
    
    # Merge basado en el año
    df_final = pd.merge(
        df_solar, 
        df_indicadores, 
        left_on='Año Instalación', 
        right_on='anio_merge', 
        how='left'
    )
    
    # Eliminar columna duplicada del merge
    if 'anio_merge' in df_final.columns:
        df_final = df_final.drop(columns=['anio_merge'])

    # Convertir textos a minusculas
    for col in df_final.select_dtypes(include='object').columns:
        df_final[col] = df_final[col].str.lower()

    # Guardar archivo enriquecido
    ruta_limpia = os.path.join("data", "energia_solar_pereira_colombia_clean.xlsx")
    df_final.to_excel(ruta_limpia, index=False)
    
    print(f"Base de datos enriquecida guardada. Columnas totales: {len(df_final.columns)}")
    return df_final
