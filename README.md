# Despliegue AI - Talento Tech

Este proyecto forma parte del diplomado de Inteligencia Artificial de Talento Tech. Consiste en una aplicacion integrada para el analisis de datos de energia solar en Pereira, Colombia, que incluye limpieza de datos, entrenamiento de modelos de clasificacion y un dashboard interactivo desarrollado con Gradio.

## Estructura del Proyecto

- `main.py`: Punto de entrada principal que orquestra la limpieza, el entrenamiento y el lanzamiento de la interfaz.
- `src/`: Carpeta con el codigo fuente dividido en modulos (procesamiento, AI, interfaces, analisis).
- `data/`: Contiene los datasets utilizados y procesados.
- `models/`: Almacena los modelos entrenados.
- `documents/`: Documentacion y reportes generados.

## Requisitos

- Python 3.8+
- Dependencias listadas en `requirements.txt` (o instaladas manualmente: pandas, scikit-learn, gradio, openpyxl, etc.)

## Uso

Para ejecutar la aplicacion completa:

```bash
python main.py
```

Esto realizara automaticamente:
1. La limpieza y el etiquetado de los datos de radiacion solar.
2. El entrenamiento del modelo de clasificacion.
3. El lanzamiento de la interfaz web interactiva con el dashboard integrado.

## Autor

**Bryan0-0AG**
