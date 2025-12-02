# Depression_Classification

El presente proyecto abarca la `clasificación de la depresión` en base a una serie de variables `en torno a los hábitos alimenticios, de higiene de sueño, de estrés, antecedentes familiares y salud mental propia`. De esta manera, tiene como objetivo discernir si el sujeto padece o no depresión, trazar posibles soluciones y sugerir herramientas de intervención.

## Estructura del Proyecto

El proyecto se estructura en base a las directivas estándares de un proyecto de Ciencia de Datos, siguiendo los pasos de obtención de datos, análisis exploratorio, limpieza de datos, modelado, predicción y presentación de resultados. 

Para conseguir lo anterior, cada etapa se trabaja en una libreta de código distinta, acompañada de un reporte final (`00_informe_final.ipynb`) donde se recapitula cada paso del proceso. 

La estructura del proyecto se puede ver aquí
```
📦 
📁 Depression_Classification/
├── 📁 assets/
│   └── 🖼️ resultados_modelos.jpg
├── 📁 data/
│   ├── 📁 interim/
│   │   ├── 📄 X_test.csv
│   │   ├── 📄 X_train.csv
│   │   └── 📄 X_val.csv
│   ├── 📁 processed/
│   │   ├── 📄 X_test.csv
│   │   ├── 📄 X_train.csv
│   │   ├── 📄 X_val.csv
│   │   ├── 📄 Y_Test.csv
│   │   ├── 📄 Y_Train.csv
│   │   └── 📄 y_val.csv
│   └── 📁 raw/
│       └── 📄 depression_dataset.csv
├── 📁 docs/
│   └── 📁 screenshots/
│       ├── 🖼️ ML_PREFECT_1.jpg
│       ├── 🖼️ ML_PREFECT_2.jpg
│       ├── 🖼️ ML_PREFECT_DATABRICKS_1.jpg
│       └── 🖼️ ML_PREFECT_DATABRICKS_2.jpg
├── 📁 notebooks/
│   ├── 📁 artifacts/
│   │   └── 📁 preprocessor/
│   │       ├── 📄 encoder.pkl
│   │       └── 📄 scaler.pkl
│   ├── 📄 00_informe_final.ipynb
│   ├── 📄 01_eda_inicial.ipynb
│   ├── 📄 02_data_wrangling.ipynb
│   └── 📄 03_training_model.ipynb
├── 📁 src/
│   ├── 📁 backend/
│   │   ├── 📁 preprocessor/
│   │   │   ├── 📄 encoder.pkl
│   │   │   └── 📄 scaler.pkl
│   │   ├── 📄 api.py
│   │   ├── 📄 Dockerfile
│   │   └── 📄 requirements.txt
│   └── 📁 frontend/
│       ├── 📄 Dockerfile
│       ├── 📄 main.py
│       └── 📄 requirements.txt
├── 📁 training_orchestration/
│   ├── 📁 artifacts/
│   │   └── 📁 preprocessor/
│   │       ├── 📄 encoder.pkl
│   │       └── 📄 scaler.pkl
│   └── 📄 Entrenamiento_Modelo.py
├── 📄 .env.example
├── 📄 .gitignore
├── 📄 .python-version
├── 📄 docker-compose.yaml
├── 📄 pyproject.toml
├── 📄 README.md
└── 📄 uv.lock


```
## 1. Manejo de Librerías y Dependencias

El manejo de librerías se lleva a cabo a través de `uv`, que necesita ser instalado para facilitar la ejecución de código y descargas de librerías. El archivo `uv.lock` contiene las librerías específicas requeridas para correr el proyecto.

## 2. Obtención de Datos

El dataset fue obtenido de Kaggle, del siguiente link: https://www.kaggle.com/datasets/hopesb/student-depression-dataset. Está descargado dentro de la carpeta `data/raw`.

## 3. Análisis Exploratorio de Datos

La libreta `01_eda_inicial.ipynb` contiene todo el código necesario para explorar los datos recopilados, abarcando un análisis por variables cualitativas y cuantitativas. La libreta ya tiene todo el código ejecutado, pero en ella se pueden encontrar funciones para facilitar el análisis por variable. 

## 4. Limpieza de Datos
La libreta `02_data_wrangling.ipynb` contiene el código utilizado para la limpieza y pre-procesamiento de los datos, incluyendo el tratamiento de valores nulos, codificación de variables categóricas y la separación entre los datasets de entrenamiento y de prueba, los cuáles se pueden observar en la carpeta de `data`.
## 5. Modelado de datos
La planeación del modeloado de datos se hizo en la libreta `03_training_model.ipynb`. La orquestración del modelado, desde la carga de datos hasta el registro de modelos utilizando Prefect, se puede encontrar en `training_orchestration/Entrenamiento_Modelo.py`. 

## 6. Backend y Frontend
El backend del proyecto (una API creada con FastAPI) se puede encontrar en el folder `src/backend` donde está el archivo .py junto con sus dependencias, los preprocesadores y el Dockerfile necesario para contenerización.

El frontend (una interfaz de usuario de Streamlit que sirve el modelo) se encuentra en el folder `src/frontend`, con los mismos componentes que tiene el backend para crear un contenedor.

En la raíz del repositorio también se puede encontrar el archivo `docker-compose.yaml` que permite ejecutar las dos partes de la aplicación en conjunto para que funcione correctamente.

# Instrucciones de Reproducción
Para reproducir el entorno de forma local, instala `uv` en tu sistema. Después, ejecutando `uv sync` en la raíz del proyecto se va a crear un ambiente virtual con la versión de Python y las dependencias necesarias para ejecutar los notebooks y archivos fuente.

Para ejecutar la aplicación en forma de contenedor se puede utilizar Docker. Utilizando Docker Compose, se puede ejecutar el comando `docker compose up --build` en la raíz del proyecto. Esto va a proporcionar una URL para probar la UI de manera local.
