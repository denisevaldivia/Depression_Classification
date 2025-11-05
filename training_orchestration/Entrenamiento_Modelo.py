# General Libraries
import os
import pandas as pd
# Databricks Env
import pathlib
import pickle
from dotenv import load_dotenv
# Feature Engineering
from sklearn.feature_extraction import DictVectorizer
# Optimization
import math
import optuna
from optuna.samplers import TPESampler
# MLFlow
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow import MlflowClient
# Modeling
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
# Evaluation Metrics
from sklearn.metrics import root_mean_squared_error
# Pipeline
from prefect import flow, task

# ======================================
# Load .env and Log in to Databricks
# ======================================

# Cargar las variables del archivo .env
load_dotenv(override=True)  
EMAIL = os.getenv('EMAIL')
PROJECT_NAME = os.getenv('PROJECT_NAME')
EXPERIMENT_NAME = f"/Users/{EMAIL}/{PROJECT_NAME}"

mlflow.set_tracking_uri("databricks")
experiment = mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)