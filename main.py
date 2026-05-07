import os
import subprocess
import sys
from modules.data_cleaning import limpiar_datos
from modules.machine_learning import entrenar_modelos_ia

def abrir_dashboard():
    print("\n🚀 Abriendo el Dashboard en Streamlit...")
    # Ejecuta 'streamlit run app.py' como un proceso separado
    try:
        # Usamos sys.executable para asegurar que use el interprete correcto
        subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py"])
    except Exception as e:
        print(f"Error al intentar abrir el dashboard: {e}")

def main():
    # Ruta del archivo original
    ruta_data = 'assets/energia_solar_pereira_colombia.xlsx'
    
    if not os.path.exists(ruta_data):
        print(f"Error: El archivo {ruta_data} no se encuentra en la carpeta assets.")
        print("Por favor, verifica la ubicacion del archivo.")
        return

    print("--- INICIANDO MODULO 1: LIMPIEZA DE DATOS ---")
    df_clean = limpiar_datos(ruta_data)

    if df_clean is not None:
        print("\n--- INICIANDO MODULO 2: DASHBOARD ---")
        abrir_dashboard()

        print("\n--- INICIANDO MODULO 3: IA ---")
        entrenar_modelos_ia(df_clean)
        
        print("\nProceso completado exitosamente.")
        print("Revisa tu navegador para ver el Dashboard interactivo.")

if __name__ == "__main__":
    main()
