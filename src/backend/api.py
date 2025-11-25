import pickle
import mlflow
from fastapi import FastAPI
from pydantic import BaseModel
from mlflow import MlflowClient
from dotenv import load_dotenv
import os
import pandas as pd
import xgboost as xgb

# Iniciar sesión en databricks
load_dotenv(override=True)  # Carga las variables del archivo .env

mlflow.set_tracking_uri("databricks")
client = MlflowClient()

EXPERIMENT_NAME = "/Users/pipochatgpt@gmail.com/Depression_Classification_prefect"

# Buscar el mejor modelo de los 3 candidatos
run_ = mlflow.search_runs(order_by=["metrics.f1 DESC"],
                          output_format="list",
                          filter_string="tags.candidate = 'true'",
                          experiment_names=[EXPERIMENT_NAME]
                          )[0]
run_id = run_.info.run_id

# Descargar artifacts (preprocessor)
client.download_artifacts(
    run_id=run_id,
    path="preprocessor",
    dst_path="."
)

with open("preprocessor/encoder.pkl", "rb") as f_in:
    encoder = pickle.load(f_in)

with open("preprocessor/scaler.pkl", "rb") as f_in:
    scaler = pickle.load(f_in)

# Cargar modelo campeón del Registry
model_name = "workspace.default.DepressionClassificationPrefect"
alias = "champion"

model_uri = f"models:/{model_name}@{alias}"
model = mlflow.pyfunc.load_model(model_uri)

# Preprocess de entrada
def preprocess(input_data):

    df = pd.DataFrame([input_data.dict()])

    # Aplicar el encoder de One Hot
    city_encoded = encoder.transform(df[["City"]])
    city_cols = encoder.get_feature_names_out(["City"])
    city_df = pd.DataFrame(city_encoded, columns=city_cols)

    # Dropear la columna de City 
    df = df.drop(columns=["City"])

    # Unir ambos datasets por columnas
    df_proc = pd.concat([df, city_df], axis=1)

    # Escalar los datos
    df_scaled = scaler.transform(df_proc)

    return df_scaled


# Realizar predicciones
def make_prediction(input_data):
    X = preprocess(input_data)
    pred = model.predict(X)
    return int(pred[0])

# FASTAPI
app = FastAPI()

# Clase de pydantic de como van los datos
class InputData(BaseModel):
    Gender: int
    Age: int
    AcademicPressure: float
    StudySatisfaction: float
    SleepDuration: int
    DietaryHabits: int
    Degree: int
    City: str
    FamilyHistory: int
    SuicidalThoughts: int

@app.post("/predict")
def predict_endpoint(input_data: InputData):
    result = make_prediction(input_data)
    return {"prediction": result}