# Estudio sobre Indicadores de Salud para la Diabetes

Este proyecto aplica técnicas de *machine learning* para analizar un conjunto de datos con indicadores de salud relacionados con la diabetes. El objetivo es predecir si una persona es saludable, prediabética o diabética, y descubrir los factores más relevantes que influyen en dicha predicción.

## Descripción del Proyecto

Utilizamos Python y bibliotecas como `NumPy`, `Pandas`, `scikit-learn`, `matplotlib`, entre otras, para:

- Realizar análisis exploratorio de los datos.
- Construir y evaluar modelos predictivos.
- Aplicar técnicas de balanceo de clases.
- Seleccionar automáticamente los atributos más relevantes.
- Comparar modelos mediante métricas y tests estadísticos.
- Interpretar resultados con herramientas explicativas como LIME o SHAP (nivel avanzado).

## Modelos Probados

- DummyClassifier (modelo base)
- Árboles de decisión
- Random Forest
- K-Nearest Neighbors
- Regresión logística
- SVM
- XGBoost (nivel avanzado)

## Técnicas Utilizadas

- Validación cruzada (10-fold CV)
- Métricas: AUC, precisión, recall, F1-score
- Balanceo de clases: Oversampling (SMOTE), Undersampling
- Selección de características: SelectKBest, RFE
- Ajuste de hiperparámetros: GridSearchCV
- Tests estadísticos: Wilcoxon signed-rank test

## Datos

- Fuente: [CDC Diabetes Health Indicators Dataset](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)
- Instancias: 253,680
- Atributos: 21
- Variable objetivo: Diagnóstico de diabetes (0: saludable, 1: prediabético/diabético)

## Requisitos

- Python 3.8+
- scikit-learn
- pandas
- matplotlib
- seaborn
- imbalanced-learn
- shap
- lime
- xgboost (opcional)

## Resultados
Los mejores resultados (AUC = 0.8262, accuracy = 0.8188, recall = 0.5678, precision = 0.3954) se alcanzaron empleando todas las variables para la predicción y RandomizedSearchCV para la búsqueda de hiperparámetros, utilizando además la estrategia de balanceo de clases mediante pesos (scale_pos_weight) con el modelo XGBoost.

## Autores
Guillermo Varela Carbajal

Raquel Gosálbez Sirvent

## Licencia
Este proyecto es parte de una práctica académica. Uso libre con fines educativos.









