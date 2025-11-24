# General Libraries
import os
import pandas as pd

# Databricks Env
import pathlib
import pickle
from dotenv import load_dotenv

# Feature Engineering
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
from xgboost import XGBClassifier

# Evaluation Metrics
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

# Pipeline
from prefect import flow, task


# ======================================
# Clean the data before processing
# ======================================

@task(name='Data Cleaning')
def clean_data(df, save_data=False):
    # 1. Eliminar valores nulos
    df = df.dropna()

    # 2. Filtrado de categorías que otorgan poca información debido a su baja prevalencia
    # City
    ciudades = df['City'].value_counts()[df['City'].value_counts() < 450]
    df = df[~df['City'].isin(ciudades.index)]
    # Dietary Habits
    df = df[df['Dietary Habits'] != 'Others']
    # Sleep Duration
    df = df[df['Sleep Duration'] != 'Others']
    # Degree
    df = df[df['Degree'] != 'Others']
    # Age
    df = df[df['Age'] <= 35]
    # Academic Pressure
    df = df[df['Academic Pressure'] > 0]
    # Study Satisfaction
    df = df[df['Study Satisfaction'] > 0]

    # 3. Eliminar variables que no son buenas predictoras
    df.drop(columns=['Work Pressure', 'Profession', 'Job Satisfaction', 'id'], axis=1, inplace=True)


    # 4. Mapear las variables categóricas binarias
    gender = {'Male' : 0, 'Female' : 1}
    general = {'Yes' : 1, 'No' : 0}
    df['Gender'] = df['Gender'].map(gender)
    df['Have you ever had suicidal thoughts ?'] = df['Have you ever had suicidal thoughts ?'].map(general)
    df['Family History of Mental Illness'] = df['Family History of Mental Illness'].map(general)


    # 5. Mapear las variables categóricas múltiples
    degree = {
    "Class 12": "Secondary",
    "B.Pharm": "Undergraduate", "BSc": "Undergraduate", "BA": "Undergraduate", "BCA": "Undergraduate",
    "B.Ed": "Undergraduate", "LLB": "Undergraduate", "BE": "Undergraduate", "BHM": "Undergraduate",
    "B.Com": "Undergraduate", "B.Arch": "Undergraduate", "B.Tech": "Undergraduate", "BBA": "Undergraduate",
    "M.Tech": "Postgraduate", "M.Ed": "Postgraduate", "MSc": "Postgraduate", "M.Pharm": "Postgraduate",
    "MCA": "Postgraduate", "MA": "Postgraduate", "MBA": "Postgraduate", "M.Com": "Postgraduate", "MHM": "Postgraduate",
    "PhD": "Doctorate", "MD": "Doctorate", "MBBS": "Doctorate", "LLM": "Doctorate", "ME": "Postgraduate"
    }
    orden_degree = {"Secondary": 0, "Undergraduate": 1, "Postgraduate": 2, "Doctorate": 3}
    orden_alimentos = {'Healthy': 0, 'Unhealthy': 1, 'Moderate': 2}
    orden_siesta = {'Less than 5 hours': 0, '5-6 hours': 1, '7-8 hours': 2,'More than 8 hours': 3}
    # Aplicar el mapeo
    df['Degree'] = df['Degree'].map(degree)
    df['Degree'] = df['Degree'].map(orden_degree)
    df['Dietary Habits'] = df['Dietary Habits'].map(orden_alimentos)
    df['Sleep Duration'] = df['Sleep Duration'].map(orden_siesta)


    # 6. Train-Test-Val Split (70-20-10)
    X = df.drop(['Depression'], axis=1)
    y = df['Depression']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.3, random_state=42)

    if save_data:
        # Guardar las variables 
        X_train.to_csv(r'..\data\interim\X_train.csv', index=False)
        X_test.to_csv(r'..\data\interim\X_test.csv', index=False)
        X_val.to_csv(r'..\data\interim\X_val.csv', index=False)
        
        y_train.to_csv(r'..\data\processed\y_train.csv', index=False)
        y_test.to_csv(r'..\data\processed\y_test.csv', index=False)
        y_val.to_csv(r'..\data\processed\y_val.csv', index=False)
    
    # Convertir las variables dependientes en NumPy arrays
    y_train = y_train.to_numpy().ravel()
    y_test = y_test.to_numpy().ravel()
    y_val = y_val.to_numpy().ravel()   

    return X_train, X_test, X_val, y_train, y_test, y_val

# ======================================
# Data wrangling and processing
# ======================================

@task(name='Data Processing')
def preprocessor(X_train, X_test, X_val=None, save_data=False, save_artifacts=True):
    # Codificar variables múltiples mediante One-Hot
    encoder = OneHotEncoder(
        drop='first',
        handle_unknown='ignore',        # Evita error si aparece algo nuevo
        sparse_output=False
    )

    # Entrenar el objeto con los datos del train
    encoder.fit(X_train[['City']])
    
    # Aplicar One-Hot
    X_train_city = encoder.transform(X_train[['City']])
    X_test_city = encoder.transform(X_test[['City']])
    X_val_city = encoder.transform(X_val[['City']]) if X_val is not None else None
    
    # Obtener los nombres del One-Hot
    city_cols = encoder.get_feature_names_out(['City'])  # Nombres automáticos de columnas
    
    # Crear un df con las columnas codificadas
    X_train_city_df = pd.DataFrame(X_train_city, columns=city_cols, index=X_train.index)
    X_test_city_df = pd.DataFrame(X_test_city, columns=city_cols, index=X_test.index)
    X_val_city_df = pd.DataFrame(X_val_city, columns=city_cols, index=X_val.index) if X_val is not None else None
    
    # Eliminar la columna original en el dataset
    X_train = X_train.drop(columns=['City'])
    X_test = X_test.drop(columns=['City'])
    X_val = X_val.drop(columns=['City']) if X_val is not None else None
    
    # Juntar las nuevas columnas con el dataset antiguo
    X_train_final = pd.concat([X_train, X_train_city_df], axis=1)
    X_test_final = pd.concat([X_test, X_test_city_df], axis=1)
    X_val_final = pd.concat([X_val, X_val_city_df], axis=1) if X_val is not None else None

    # Aplicar una estandarización a los datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_final)
    X_test_scaled = scaler.transform(X_test_final)
    X_val_scaled = scaler.transform(X_val_final) if X_val is not None else None

    # Guardar los artefactos
    if save_artifacts:
        os.makedirs("artifacts/preprocessor", exist_ok=True)

        # Save encoder
        with open('artifacts/preprocessor/encoder.pkl', 'wb') as f_out:
            pickle.dump(encoder, f_out)
        # Save scaler
        with open('artifacts/preprocessor/scaler.pkl', 'wb') as f_out:
            pickle.dump(scaler, f_out)

        # Log artifacts to MLflow
        mlflow.log_artifact("artifacts/preprocessor/encoder.pkl", artifact_path="preprocessor")
        mlflow.log_artifact("artifacts/preprocessor/scaler.pkl", artifact_path="preprocessor")

        print("Preprocessor artifacts (encoder & scaler) successfully logged to MLflow.")

    if save_data:
        # Regresar los datos a dataframe y guardarlos
        X_train_df = pd.DataFrame(X_train_scaled, columns=X_train_final.columns, index=X_train_final.index)
        X_test_df = pd.DataFrame(X_test_scaled, columns=X_test_final.columns, index=X_test_final.index)
        X_val_df = pd.DataFrame(X_val_scaled, columns=X_val_final.columns, index=X_val_final.index) if X_val is not None else None

        X_train_df.to_csv(r'..\data\processed\X_train.csv', index=False)
        X_test_df.to_csv(r'..\data\processed\X_test.csv', index=False)
        X_val_df.to_csv(r'..\data\processed\X_val.csv', index=False)

    return X_train_scaled, X_test_scaled, X_val_scaled, encoder, scaler

# ======================================
# Find best parameters for models
# ======================================

@task(name = 'Hyperparameter Tuning - LR')
def hp_tuning_lr(X_train, X_test, y_train, y_test):

    mlflow.sklearn.autolog()

    # Start Optuna and MLflow
    def objective_lr(trial: optuna.trial.Trial):
        params = {
            'penalty': trial.suggest_categorical('penalty', ['l2','l1','elasticnet']),
            'solver': 'saga'
        }

        with mlflow.start_run(nested=True):
            # Preprocess data and log artifacts
            X_train_scaled, X_test_scaled, _, encoder, scaler = preprocessor(X_train, X_test, X_val=None, save_artifacts=True)

            # Get MLflow ID to store the preprocessing artifacts
            preprocessor_run_id = mlflow.active_run().info.run_id

            mlflow.set_tag('model_family', 'logistic_regression')
            mlflow.log_params(params)
            mlflow.log_param('preprocessor_run_id', preprocessor_run_id)

            lr_model = LogisticRegression(**params)
            lr_model.fit(X_train_scaled, y_train)

            # Get predictions and metrics
            y_pred = lr_model.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

            # Log metrics
            mlflow.log_metric('acc', acc)
            mlflow.log_metric('precision', precision)
            mlflow.log_metric('f1', f1)
            mlflow.log_metric('recall', recall)

            signature = infer_signature(X_test_scaled, y_pred)

            # Log the trained model
            mlflow.sklearn.log_model(
                lr_model,
                name='lr_model',
                input_example=X_test_scaled[:5],
                signature=signature
            )
        
        return f1
    
    sampler = TPESampler(seed=42)
    lr_study = optuna.create_study(direction='maximize', sampler=sampler)

    with mlflow.start_run(run_name='Logistic Regression (Optuna)', nested=True):
        lr_study.optimize(objective_lr, n_trials=3)
    
    best_params_lr = lr_study.best_params

    return best_params_lr

@task(name = 'Hyperparameter Tuning - SVC')
def hp_tuning_svc(X_train, X_test, y_train, y_test):

    mlflow.sklearn.autolog()

    def objective_svc(trial: optuna.trial.Trial):
        params = {
            'kernel': trial.suggest_categorical('kernel', ['sigmoid','poly','linear','rbf'])
        }

        with mlflow.start_run(nested=True):
            # Preprocess data and log artifacts
            X_train_scaled, X_test_scaled, _, encoder, scaler = preprocessor(X_train, X_test, X_val=None, save_artifacts=True)

            # Get MLflow ID to store the preprocessing artifacts
            preprocessor_run_id = mlflow.active_run().info.run_id

            mlflow.set_tag('model_family', 'svc')
            mlflow.log_params(params)
            mlflow.log_param('preprocessor_run_id', preprocessor_run_id)

            svc_model = SVC(**params)
            svc_model.fit(X_train_scaled, y_train)

            y_pred = svc_model.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

            mlflow.log_metric('acc', acc)
            mlflow.log_metric('precision', precision)
            mlflow.log_metric('f1', f1)
            mlflow.log_metric('recall', recall)

            signature = infer_signature(X_test_scaled, y_pred)

            mlflow.sklearn.log_model(
                svc_model,
                name='svc_model',
                input_example=X_test_scaled[:5],
                signature=signature
            )
        
        return f1

    
    sampler = TPESampler(seed=42)
    svc_study = optuna.create_study(direction='maximize', sampler=sampler)

    with mlflow.start_run(run_name='Support Vector Classifier (Optuna)', nested=True):
        svc_study.optimize(objective_svc, n_trials=3)
    
    best_params_svc = svc_study.best_params

    best_params_svc['random_state'] = 42

    return best_params_svc

@task(name = 'Hyperparameter Tuning - XGBoost')
def hp_tuning_xgboost(X_train, X_test, y_train, y_test):
    # Habilitar autolog
    mlflow.xgboost.autolog()

    # Función objetivo para Optuna
    def objective_xgb(trial: optuna.trial.Trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 150),
            'max_depth': trial.suggest_int('max_depth', 2, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'eval_metric': 'logloss'
        }

        with mlflow.start_run(nested=True):
            # Preprocess data and log artifacts
            X_train_scaled, X_test_scaled, _, encoder, scaler = preprocessor(X_train, X_test, X_val=None, save_artifacts=True)

            # Get MLflow ID to store the preprocessing artifacts
            preprocessor_run_id = mlflow.active_run().info.run_id

            mlflow.set_tag('model_family', 'Xgboost')
            mlflow.log_params(params)
            mlflow.log_param('preprocessor_run_id', preprocessor_run_id)

            xgb_model = XGBClassifier(**params)
            xgb_model.fit(X_train_scaled, y_train)

            y_pred = xgb_model.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

            mlflow.log_metric('acc', acc)
            mlflow.log_metric('precision', precision)
            mlflow.log_metric('f1', f1)
            mlflow.log_metric('recall', recall)

            signature = infer_signature(X_test_scaled, y_pred)

            mlflow.xgboost.log_model(
                xgb_model,
                artifact_path='xgboost_model',
                input_example=X_test_scaled[:5],
                signature=signature
            )
        
        return f1

    # Crear y ejecutar el estudio de Optuna
    sampler = TPESampler(seed=42)
    xgb_study = optuna.create_study(direction='maximize', sampler=sampler)

    with mlflow.start_run(run_name='XGBoost (Optuna)', nested=True):
        xgb_study.optimize(objective_xgb, n_trials=3)

    # Obtener los mejores parámetros
    best_params_xgb = xgb_study.best_params
    best_params_xgb['random_state'] = 42

    return best_params_xgb

# ======================================
# Train best models
# ======================================

@task(name='Train Models')
def train_best_models(X_train, y_train, X_test, y_test, best_params_lr, best_params_svc, best_params_xgb) -> None:
    with mlflow.start_run(run_name=' Best Logistic Regression Model'):
        # Preprocess data and log artifacts
        X_train_scaled, X_test_scaled, _, encoder, scaler = preprocessor(X_train, X_test, X_val=None, save_artifacts=True)

        # Get MLflow ID to store the preprocessing artifacts
        preprocessor_run_id = mlflow.active_run().info.run_id
        mlflow.log_param('preprocessor_run_id', preprocessor_run_id)
        mlflow.log_params(best_params_lr)
        mlflow.set_tags({
            'project': 'Depression Prediction Project',
            'optimizer_engine': 'Optuna',
            'model_family': 'logistic_regression',
            'feature_set_version': 1,
            'candidate': 'true'
        })

        lr = LogisticRegression(**best_params_lr, solver='saga')
        lr.fit(X_train_scaled, y_train)

        y_pred_lr = lr.predict(X_test_scaled)

        acc_lr = accuracy_score(y_test, y_pred_lr)
        precision_lr = precision_score(y_test, y_pred_lr)
        f1_lr = f1_score(y_test, y_pred_lr)
        recall_lr = recall_score(y_test, y_pred_lr)

        mlflow.log_metric('acc', acc_lr)
        mlflow.log_metric('precision', precision_lr)
        mlflow.log_metric('f1', f1_lr)
        mlflow.log_metric('recall', recall_lr)

        signature = infer_signature(X_train_scaled, lr.predict(X_train_scaled))
        mlflow.sklearn.log_model(
            lr,
            artifact_path='model',
            signature=signature
        )
    
    with mlflow.start_run(run_name=' Best SVC Model'):
        # Preprocess data and log artifacts
        X_train_scaled, X_test_scaled, _, encoder, scaler = preprocessor(X_train, X_test, X_val=None, save_artifacts=True)

        # Get MLflow ID to store the preprocessing artifacts
        preprocessor_run_id = mlflow.active_run().info.run_id
        mlflow.log_param('preprocessor_run_id', preprocessor_run_id)
        mlflow.log_params(best_params_svc)
        mlflow.set_tags({
            'project': 'Depression Prediction Project',
            'optimizer_engine': 'Optuna',
            'model_family': 'svc',
            'feature_set_version': 1,
            'candidate': 'true'
        })

        svc = SVC(**best_params_svc)
        svc.fit(X_train_scaled, y_train)

        y_pred_svc = svc.predict(X_test_scaled)

        acc_svc = accuracy_score(y_test, y_pred_svc)
        precision_svc = precision_score(y_test, y_pred_svc)
        f1_svc = f1_score(y_test, y_pred_svc)
        recall_svc = recall_score(y_test, y_pred_svc)

        mlflow.log_metric('acc', acc_svc)
        mlflow.log_metric('precision', precision_svc)
        mlflow.log_metric('f1', f1_svc)
        mlflow.log_metric('recall', recall_svc)

        signature = infer_signature(X_train_scaled, svc.predict(X_train_scaled))
        mlflow.sklearn.log_model(
            svc,
            artifact_path='model',
            signature=signature
        )
    
    with mlflow.start_run(run_name=' Best XGBoost Model'):
        # Preprocess data and log artifacts
        X_train_scaled, X_test_scaled, _, encoder, scaler = preprocessor(X_train, X_test, X_val=None, save_artifacts=True)

        # Get MLflow ID to store the preprocessing artifacts
        preprocessor_run_id = mlflow.active_run().info.run_id
        mlflow.log_param('preprocessor_run_id', preprocessor_run_id)
        mlflow.log_params(best_params_xgb)
        mlflow.set_tags({
            'project': 'Depression Prediction Project',
            'optimizer_engine': 'Optuna',
            'model_family': 'Trees',
            'feature_set_version': 1,
            'candidate': 'true'
        })

        xgb = XGBClassifier(**best_params_xgb)
        xgb.fit(X_train_scaled, y_train)
        y_pred_xgb = xgb.predict(X_test_scaled)

        acc_xgb = accuracy_score(y_test, y_pred_xgb)
        precision_xgb = precision_score(y_test, y_pred_xgb)
        f1_xgb = f1_score(y_test, y_pred_xgb)
        recall_xgb = recall_score(y_test, y_pred_xgb)

        mlflow.log_metric('acc', acc_xgb)
        mlflow.log_metric('precision', precision_xgb)
        mlflow.log_metric('f1', f1_xgb)
        mlflow.log_metric('recall', recall_xgb)

        signature = infer_signature(X_train_scaled, xgb.predict(X_train_scaled))
        mlflow.xgboost.log_model(
            xgb,
            artifact_path='model',
            signature=signature
        )

# ======================================
# Register Models
# ======================================

@task(name='Model Registry')
def register_champion_challenger(exp, model_registry_name="workspace.default.DepressionClassPrefect"):
    client = MlflowClient()

    # Buscar los runs candidatos ordenados por F1
    runs = mlflow.search_runs(
        experiment_names=[exp],
        filter_string="tags.candidate = 'true'",
        order_by=["metrics.f1 DESC"]
    )

    if runs.empty:
        print("No candidate runs found.")
        return

    # Tomar los dos mejores
    champion = runs.iloc[0]
    challenger = runs.iloc[1] if len(runs) > 1 else None

    def register(run_row, alias):
        if run_row is None:
            return

        run_id = run_row['run_id']
        f1 = run_row['metrics.f1']
        model_family = run_row['tags.model_family']

        # Registrar modelo
        result = mlflow.register_model(
            model_uri=f"runs:/{run_id}/model",
            name=model_registry_name
        )

        # Asignar alias
        client.set_registered_model_alias(
            name=model_registry_name,
            alias=alias,
            version=result.version
        )

        print(f"{alias} registrado: {model_family} con F1={f1} (Run ID: {run_id})")

    register(champion, "Champion")
    register(challenger, "Challenger")

@flow(name="Main Flow")
def main_flow() -> None:
    """The main training pipeline"""
    
    # Load .env and Log in to Databricks
    load_dotenv(override=True)  # Carga las variables del archivo .env
    EXPERIMENT_NAME = "/Users/pipochatgpt@gmail.com/Depression_Class_prefect"

    mlflow.set_tracking_uri("databricks")
    experiment = mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)

    # Load Data
    df = pd.read_csv("../data/raw/depression_dataset.csv")

    # Clean the data and obtain the targets
    X_train, X_test, X_val, y_train, y_test, y_val = clean_data(df)
    
    # Hyper-parameter Tunning
    best_params_xgb = hp_tuning_xgboost(X_train, X_test, y_train, y_test)
    best_params_svc = hp_tuning_svc(X_train, X_test, y_train, y_test)
    best_params_lr = hp_tuning_lr(X_train, X_test, y_train, y_test)
    
    # Train Models
    train_best_models(X_train, y_train, X_test, y_test, best_params_lr, best_params_svc, best_params_xgb)

    # Setear la URI del Model Registry a legacy Workspace
    mlflow.set_registry_uri("databricks-uc")
    register_champion_challenger(exp=EXPERIMENT_NAME)

if __name__ == "__main__":
    main_flow()

