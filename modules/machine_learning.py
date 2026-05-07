import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

def calcular_metricas(y_real, y_pred):
    mae = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    r2 = r2_score(y_real, y_pred)
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2:", r2)
    return mae, rmse, r2

def graficar_resultados(y_test, y_pred, nombre_modelo):
    # Reales vs Predichos
    y_real_arr = np.array(y_test)
    y_pred_arr = np.array(y_pred)
    n = min(40, len(y_real_arr))
    indices = np.arange(n)

    plt.figure(figsize=(10, 5))
    plt.plot(indices, y_real_arr[:n], marker='o', linestyle='-', color='red', label='Valores reales')
    plt.plot(indices, y_pred_arr[:n], marker='o', linestyle='-', color='blue', label='Valores predichos')
    plt.title(f"{nombre_modelo}: valores reales y predichos")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    # Residuos
    residuos = y_real_arr - y_pred_arr
    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred_arr, residuos, color='mediumpurple', alpha=0.75, edgecolors='black')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title(f"{nombre_modelo}: diagrama de residuos")
    plt.show()

    # Histograma Residuos
    plt.figure(figsize=(8, 5))
    plt.hist(residuos, bins=25, color='mediumpurple', edgecolor='black', alpha=0.8)
    plt.axvline(x=0, color='red', linestyle='--')
    plt.title(f"{nombre_modelo}: histograma de residuos")
    plt.show()

def entrenar_modelos_ia(df):
    if df is None:
        return

    # 1. Separar X e y
    X = df.drop('Factura Ahorrada Mensual', axis=1)
    y = df['Factura Ahorrada Mensual']

    # 2. Definir tipos de variables
    numericas = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    nominales = X.select_dtypes(include=['object', 'string']).columns.tolist()

    # 3. Crear preprocesador
    preprocesador = ColumnTransformer(
        transformers=[
            ('nom', OneHotEncoder(drop='first', handle_unknown='ignore'), nominales),
            ('num', 'passthrough', numericas)
        ]
    )

    # 4. Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Transformar datos
    X_train_pre = preprocesador.fit_transform(X_train)
    X_test_pre = preprocesador.transform(X_test)

    # --- Modelos ---
    
    # Arbol de Decision
    print("\nResultados - Arbol de Decision")
    modelo_arbol = DecisionTreeRegressor(random_state=42)
    modelo_arbol.fit(X_train_pre, y_train)
    y_pred_arbol = modelo_arbol.predict(X_test_pre)
    mae_arbol, rmse_arbol, r2_arbol = calcular_metricas(y_test, y_pred_arbol)
    graficar_resultados(y_test, y_pred_arbol, "Arbol de Decision")

    # Random Forest
    print("\nResultados - Random Forest")
    modelo_rf = RandomForestRegressor(random_state=42, n_estimators=100)
    modelo_rf.fit(X_train_pre, y_train)
    y_pred_rf = modelo_rf.predict(X_test_pre)
    mae_rf, rmse_rf, r2_rf = calcular_metricas(y_test, y_pred_rf)
    graficar_resultados(y_test, y_pred_rf, "Random Forest")

    # Regresion Lineal
    print("\nResultados - Regresion Lineal")
    modelo_lineal = LinearRegression()
    modelo_lineal.fit(X_train_pre, y_train)
    y_pred_lineal = modelo_lineal.predict(X_test_pre)
    mae_lineal, rmse_lineal, r2_lineal = calcular_metricas(y_test, y_pred_lineal)
    graficar_resultados(y_test, y_pred_lineal, "Regresion Lineal")

    # XGBoost
    print("\nResultados - XGBoost")
    modelo_xgb = XGBRegressor(
        random_state=42, n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, objective='reg:squarederror'
    )
    modelo_xgb.fit(X_train_pre, y_train)
    y_pred_xgb = modelo_xgb.predict(X_test_pre)
    mae_xgb, rmse_xgb, r2_xgb = calcular_metricas(y_test, y_pred_xgb)
    graficar_resultados(y_test, y_pred_xgb, "XGBoost")

    # Comparacion final
    resultados = pd.DataFrame({
        'Modelo': ['Arbol de Decision', 'Random Forest', 'Regresion Lineal', 'XGBoost'],
        'MAE': [mae_arbol, mae_rf, mae_lineal, mae_xgb],
        'RMSE': [rmse_arbol, rmse_rf, rmse_lineal, rmse_xgb],
        'R2': [r2_arbol, r2_rf, r2_lineal, r2_xgb]
    })
    print("\nComparacion de modelos:")
    print(resultados)
    return resultados
