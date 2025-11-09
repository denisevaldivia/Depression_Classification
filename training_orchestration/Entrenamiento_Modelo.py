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
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
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

# ======================================
# Read processed data
# ======================================

@task(name='Data Reading')
def read_data(X_train_path, Y_train_path, X_test_path, Y_test_path):
    X_train = pd.read_csv(X_train_path)
    Y_train = pd.read_csv(Y_train_path)
    X_test = pd.read_csv(X_test_path)
    Y_test = pd.read_csv(Y_test_path)

    return X_train, Y_train, X_test, Y_test

# ======================================
# Find best parameters for models
# ======================================

@task(name = 'Hyperparameter Tuning - LR')
def hp_tuning_lr(X_train, X_test, Y_train, Y_test):

    mlflow.sklearn.autolog()

    training_dataset = mlflow.data.from_numpy(X_train.data, targets=Y_train, name='Train Data')
    validation_dataset = mlflow.data.from_numpy(X_test.data, targets=Y_test, name='Test Data')

    def objective_lr(trial: optuna.trial.Trial):
        params = {
            'penalty': trial.suggest_categorical('penalty', ['l2','l1','elasticnet'])
        }

        with mlflow.start_run(nested=True):
            mlflow.set_tag('model_family', 'logistic_regression')
            mlflow.log_params(params)

            lr_model = LogisticRegression(**params)
            lr_model.fit(X_train, Y_train)

            y_pred = lr_model.predict(X_test)
            acc = accuracy_score(Y_test, y_pred)
            precision = precision_score(Y_test, y_pred)
            f1 = f1_score(Y_test, y_pred)
            recall = recall_score(Y_test, y_pred)

            mlflow.log_metric('acc', acc)
            mlflow.log_metric('precision', precision)
            mlflow.log_metric('f1', f1)
            mlflow.log_metric('recall', recall)

            signature = infer_signature(X_test, y_pred)

            mlflow.sklearn.log_model(
                lr_model,
                name='lr_model',
                input_example=X_test[:5],
                signature=signature
            )
        
        return acc
    
    sampler = TPESampler(seed=42)
    lr_study = optuna.create_study(direction='maximize', sampler=sampler)

    with mlflow.start_run(run_name='Logisitc Regression (Optuna)', nested=True):
        lr_study.optimize(objective_lr, n_trials=3)
    
    best_params_lr = lr_study.best_params

    return best_params_lr

@task(name = 'Hyperparameter Tuning - SVC')
def hp_tuning_svc(X_train, X_test, Y_train, Y_test):

    mlflow.sklearn.autolog()

    training_dataset = mlflow.data.from_numpy(X_train.data, targets=Y_train, name='Train Data')
    validation_dataset = mlflow.data.from_numpy(X_test.data, targets=Y_test, name='Test Data')

    def objective_svc(trial: optuna.trial.Trial):
        params = {
            'kernel': trial.suggest_categorical('kernel', ['sigmoid','poly','linear','rbf'])
        }

        with mlflow.start_run(nested=True):
            mlflow.set_tag('model_family', 'svc')
            mlflow.log_params(params)

            svc_model = SVC(**params)
            svc_model.fit(X_train, Y_train)

            y_pred = svc_model.predict(X_test)
            acc = accuracy_score(Y_test, y_pred)
            precision = precision_score(Y_test, y_pred)
            f1 = f1_score(Y_test, y_pred)
            recall = recall_score(Y_test, y_pred)

            mlflow.log_metric('acc', acc)
            mlflow.log_metric('precision', precision)
            mlflow.log_metric('f1', f1)
            mlflow.log_metric('recall', recall)

            signature = infer_signature(X_test, y_pred)

            mlflow.sklearn.log_model(
                svc_model,
                name='svc_model',
                input_example=X_test[:5],
                signature=signature
            )
        
        return acc
    
    sampler = TPESampler(seed=42)
    svc_study = optuna.create_study(direction='maximize', sampler=sampler)

    with mlflow.start_run(run_name='Support Vector Classifier (Optuna)', nested=True):
        svc_study.optimize(objective_svc, n_trials=3)
    
    best_params_svc = svc_study.best_params

    best_params_svc['random_state'] = 42

    return best_params_svc

# ======================================
# Train best models
# ======================================

@task(name='Train Models')
def train_best_models(X_train, Y_train, X_test, Y_test, best_params_lr, best_params_svc) -> None:
    with mlflow.start_run(run_name='Logistic Regression Model'):
        mlflow.log_params(best_params_lr)
        mlflow.set_tags({
            'project': 'Depression Prediction Project',
            'optimizer_engine': 'Optuna',
            'model_family': 'logistic_regression',
            'feature_set_version': 1
        })

        lr = LogisticRegression(**best_params_lr)
        lr.fit(X_train, Y_train)

        y_pred_lr = lr.predict(X_test)

        acc_lr = accuracy_score(Y_test, y_pred_lr)
        precision_lr = precision_score(Y_test, y_pred_lr)
        f1_lr = f1_score(Y_test, y_pred_lr)
        recall_lr = recall_score(Y_test, y_pred_lr)

        mlflow.log_metric('acc', acc_lr)
        mlflow.log_metric('precision', precision_lr)
        mlflow.log_metric('f1', f1_lr)
        mlflow.log_metric('recall', recall_lr)

        mlflow.sklearn.log_model(
            lr,
            name='model'
        )
    
    with mlflow.start_run(run_name='SVC Model'):
        mlflow.log_params(best_params_svc)
        mlflow.set_tags({
            'project': 'Depression Prediction Project',
            'optimizer_engine': 'Optuna',
            'model_family': 'svc',
            'feature_set_version': 1
        })

        svc = SVC(**best_params_svc)
        svc.fit(X_train, Y_train)

        y_pred_svc = svc.predict(X_test)

        acc_svc = accuracy_score(Y_test, y_pred_svc)
        precision_svc = precision_score(Y_test, y_pred_svc)
        f1_svc = f1_score(Y_test, y_pred_svc)
        recall_svc = recall_score(Y_test, y_pred_svc)

        mlflow.log_metric('acc', acc_svc)
        mlflow.log_metric('precision', precision_svc)
        mlflow.log_metric('f1', f1_svc)
        mlflow.log_metric('recall', recall_svc)

        mlflow.sklearn.log_model(
            lr,
            name='model'
        )

