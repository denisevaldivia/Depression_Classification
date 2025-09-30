# Depression_Classification

El presente proyecto abarca la `clasificación de la depresión` en base a una serie de variables `en torno a los hábitos alimenticios, de higiene de sueño, de estrés, antecedentes familiares y salud mental propia`. De esta manera, tiene como objetivo discernir si el sujeto padece o no depresión, trazar posibles soluciones y sugerir herramientas de intervención.

## Estructura del Proyecto

El proyecto se estructura en base a las directivas estándares de un proyecto de Ciencia de Datos, siguiendo los pasos de obtención de datos, análisis exploratorio, limpieza de datos, modelado, predicción y presentación de resultados. 

Para conseguir lo anterior, cada etapa se trabaja en una libreta de código distinta, acompañada de un reporte final (`00_informe_final.ipynb`) donde se recapitula cada paso del proceso. 

## 1. Manejo de Librerías y Dependencias

El manejo de librerías se lleva a cabo a través de `uv`, que necesita ser instalado para facilitar la ejecución de código y descargas de librerías. El archivo `uv.lock` contiene las librerías específicas requeridas para correr el proyecto.

## 2. Obtención de Datos

El dataset fue obtenido de Kaggle, del siguiente link: https://www.kaggle.com/datasets/hopesb/student-depression-dataset. Está descargado dentro de la carpeta data/raw.

## 3. Análisis Exploratorio de Datos

La libreta `01_eda_inicial.ipynb` contiene todo el código necesario para explorar los datos recopilados, abarcando un análisis por variables cualitativas y cuantitativas. La libreta ya tiene todo el código ejecutado, pero en ella se pueden encontrar funciones para facilitar el análisis por variable. 

## 4. Limpieza de Datos