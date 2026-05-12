import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, probplot, shapiro

def obtener_graficos_categoricos(df):
    figs = []
    columnas = ['Año Instalación', 'Barrio/Sector', 'Material Panel', 'Tipo', 'Empresa Instaladora', 'Estado']
    for col in columnas:
        if col in df.columns:
            valor = df[col].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(valor, labels=valor.index, autopct="%1.1f%%", colors=plt.cm.Paired.colors)
            ax.set_title(f"Distribucion de {col}")
            figs.append(fig)
    return figs

def obtener_graficos_numericos(df):
    figs = []
    df_cuant = df.select_dtypes(include=['int64', 'float64'])
    for columna in df_cuant.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df_cuant[columna], kde=True, bins=20, stat="density", color="blue", label="Datos", ax=ax)
        media = df_cuant[columna].mean()
        desviacion = df_cuant[columna].std()
        x = np.linspace(df_cuant[columna].min(), df_cuant[columna].max(), 100)
        y = norm.pdf(x, media, desviacion)
        ax.plot(x, y, color="red", label="Campana de Gauss")
        ax.set_title(f"Distribucion de {columna}")
        ax.grid(True)
        figs.append(fig)
    return figs

def obtener_graficos_analisis_target(df, columna_target="Factura Ahorrada Mensual"):
    figs = []
    if columna_target in df.columns:
        datos = df[columna_target].dropna()
        fig_qq, ax_qq = plt.subplots(figsize=(6,6))
        probplot(datos, dist="norm", plot=ax_qq)
        ax_qq.set_title(f"QQ-plot - {columna_target}")
        figs.append(fig_qq)
        fig_box, ax_box = plt.subplots(figsize=(8,2.5))
        ax_box.boxplot(datos, vert=False)
        ax_box.set_title(f"Boxplot - {columna_target}")
        figs.append(fig_box)
    return figs

def obtener_graficos_correlacion(df):
    df_cuant = df.select_dtypes(include=['int64', 'float64'])
    correlacion = df_cuant.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(correlacion, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Matriz de Correlacion")
    return fig

def obtener_grafico_nulos(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(df.isnull(), cmap="viridis", cbar=False, yticklabels=False, ax=ax)
    ax.set_title("Mapa de calor de valores nulos")
    return fig
