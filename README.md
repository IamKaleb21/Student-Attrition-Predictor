
# Minería de Datos: Student-Attrition-Predictor

## Descripción

Este proyecto tiene como objetivo predecir la deserción estudiantil utilizando técnicas de minería de datos. Para lograrlo, se parte de un conjunto de datos de estudiantes y se realiza un análisis exploratorio, limpieza de datos, creación de características y la aplicación de algoritmos de aprendizaje automático para identificar patrones que permitan predecir la deserción.

El enfoque incluye la preparación de datos, tratamiento de valores nulos, eliminación de duplicados y generación de nuevas características que ayuden a mejorar la predicción.

## Tabla de Contenidos

1. [Instalación](#instalación)
2. [Uso](#uso)
3. [Fases del Proyecto](#fases-del-proyecto)
4. [Características](#características)
5. [Dependencias](#dependencias)
6. [Contribuidores](#contribuidores)
7. [Licencia](#licencia)

## Instalación

1. Clonar este repositorio en tu máquina local.
2. Instalar las dependencias necesarias. Puedes hacerlo utilizando `pip`:

    ```bash
    pip install -r requirements.txt
    ```

Si no tienes un archivo `requirements.txt`, estas son las principales bibliotecas que debes instalar:

```bash
bashpip install pandas numpy scikit-learn matplotlib
```

3. Asegúrate de tener Python 3.x instalado en tu sistema.

## Uso

1. Cargar los datos de estudiantes desde un archivo CSV:

```python
pythonimport pandas as pd

data = pd.read_csv('estudiantes.csv')
```

2. Realizar la limpieza de datos, como eliminar duplicados y rellenar valores nulos:

```python
python# Eliminar duplicados
data = data.drop_duplicates()

# Imputación de valores nulos
data['Creditos_Curso'].fillna(data['Creditos_Curso'].median(), inplace=True)
data['Num_Vez'].fillna(data['Num_Vez'].median(), inplace=True)
```

3. Generar características adicionales que sean útiles para los modelos predictivos y explorar los datos:

```python
python# Generar nuevas características
data['Nueva_Caracteristica'] = data['Creditos_Curso'] * data['Num_Vez']
```

4. Aplicar modelos de aprendizaje automático para predecir la deserción estudiantil. Por ejemplo, puedes usar un modelo de regresión logística:

```python
pythonfrom sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Dividir los datos en entrenamiento y prueba
X = data[['Creditos_Curso', 'Num_Vez']]
y = data['Desercion']  # Asumiendo que hay una columna 'Desercion' que es la variable a predecir
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo
modelo = LogisticRegression()
modelo.fit(X_train, y_train)

# Evaluar el modelo
accuracy = modelo.score(X_test, y_test)
print(f"Precisión del modelo: {accuracy}")
```

## Fases del Proyecto

### 1\. Preparación de los Datos

Se realiza una serie de pasos para asegurar que los datos estén en las mejores condiciones posibles para ser utilizados en los modelos predictivos. Esto incluye:

* **Eliminación de duplicados**.
* **Manejo de valores nulos**: se imputan valores utilizando la mediana para algunas variables.
* **Normalización y transformación de características**.

### 2\. Análisis Exploratorio de Datos (EDA)

Durante esta fase, se exploran los datos para entender la distribución de las variables, la relación entre las variables predictoras y la variable objetivo (deserción).

### 3\. Generación de Características

Se crean nuevas características basadas en los datos existentes, las cuales pueden mejorar la capacidad predictiva del modelo.

### 4\. Entrenamiento del Modelo

Se selecciona y entrena un modelo de aprendizaje automático (como regresión logística, árboles de decisión, etc.) para predecir la deserción estudiantil.

### 5\. Evaluación del Modelo

Finalmente, se evalúa el modelo utilizando métricas de precisión y se ajusta según sea necesario.

## Características

* **Análisis de deserción estudiantil**: Permite predecir la probabilidad de que un estudiante deserte.
* **Procesamiento de datos**: Incluye técnicas de limpieza y generación de características.
* **Modelado predictivo**: Se entrenan y evalúan varios modelos de clasificación para seleccionar el más adecuado.

## Dependencias

Este proyecto utiliza las siguientes bibliotecas:

* **pandas**: Para la manipulación y análisis de datos.
* **numpy**: Para operaciones matemáticas y matrices.
* **scikit-learn**: Para el entrenamiento de modelos de machine learning.
* **matplotlib**: Para la visualización de datos.

## Contribuidores

* Aarón Kaleb Arteaga Rodríguez

## Licencia

Este proyecto está licenciado bajo los términos de la Licencia MIT.


