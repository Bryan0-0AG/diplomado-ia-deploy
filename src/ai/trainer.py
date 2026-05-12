import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

def entrenar_modelo_clasificacion(df):
    # Columnas tecnicas y economicas añadidas
    columnas_entrada = [
        'Año Instalación', 'Tipo', 'Material Panel', 'N° Paneles', 'Radiación Solar',
        'Eficiencia Panel (%)', 'Humedad Relativa Prom', 'Temperatura Prom',
        'ipc_general_pct', 'ipc_energia_pct', 'trm_promedio_cop'
    ]
    target = 'Factura Ahorrada Mensual'

    # Asegurar que no hay nulos en las columnas seleccionadas
    df = df.dropna(subset=columnas_entrada + [target])

    X = df[columnas_entrada]
    y = df[target]

    # Preprocesamiento
    categoricas = ['Tipo', 'Material Panel']
    numericas = [
        'Año Instalación', 'N° Paneles', 'Radiación Solar', 
        'Eficiencia Panel (%)', 'Humedad Relativa Prom', 'Temperatura Prom',
        'ipc_general_pct', 'ipc_energia_pct', 'trm_promedio_cop'
    ]

    preprocesador = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categoricas),
            ('num', StandardScaler(), numericas)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train_pre = preprocesador.fit_transform(X_train)
    X_test_pre = preprocesador.transform(X_test)

    # Entrenar Regresor con hiperparametros base
    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X_train_pre, y_train)

    preds = modelo.predict(X_test_pre)
    r2 = r2_score(y_test, preds)
    print(f"Modelo enriquecido entrenado con R2 Score: {r2:.4f}")

    os.makedirs('models', exist_ok=True)
    joblib.dump(modelo, 'models/modelo_solar.pkl')
    joblib.dump(preprocesador, 'models/preprocesador.pkl')
    
    return modelo, preprocesador
