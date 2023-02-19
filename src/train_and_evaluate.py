import os
import yaml
import pandas as pd
import argparse
from pkgutil import get_data
from get_data import get_data,read_params
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from xgboost import XGBClassifier
import joblib
import json
import numpy as np
import mlflow
from urllib.parse import urlparse


def eval_metrics(actual,pred):
    accuracy = accuracy_score(actual,pred)
    precision_scr = precision_score(actual,pred)
    recall_scr = recall_score(actual,pred)
    f1_score = recall_score(actual,pred)
    return accuracy,precision_scr,recall_scr,f1_score



def train_and_evaluate(config_path):
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    random_state = config["base"]["random_state"]
    model_dir = config["model_dir"]
    max_depth = config["estimators"]["XGBClassifier"]["params"]["max_depth"]
    booster = config["estimators"]["XGBClassifier"]["params"]["booster"]
    target = config["base"]["target_col"]
    train=pd.read_csv(train_data_path,sep=",")
    test=pd.read_csv(test_data_path,sep=",")
    
    train_x = train.drop(target,axis=1)
    test_x = test.drop(target,axis=1)
    train_y = train[target]
    test_y = test[target]



    ########################################################

    mlflow_config = config["mlflow_config"]
    remote_server_uri = mlflow_config["remote_server_uri"]
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(mlflow_config["experiment_name"])
    
    with mlflow.start_run(run_name= mlflow_config["run_name"]) as mlops_run:
        lr = XGBClassifier(max_depth = max_depth,random_state=random_state,booster=booster)
        lr.fit(np.array(train_x),np.array(train_y))
        predicted_qualities = lr.predict(np.array(test_x))
        accuracy,precison_src,recall_scr,f1_score= eval_metrics(test_y,predicted_qualities)
        mlflow.log_param("max_depth",max_depth)
        mlflow.log_param("booster",booster)

        mlflow.log_metric("accuracy",accuracy)
        mlflow.log_metric("F1_Score",f1_score)
        mlflow.log_metric("Precision",precison_src)
        mlflow.log_metric("Recall",recall_scr)


        tracting_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

        if tracting_url_type_store !="file":
            mlflow.sklearn.log_model(lr,"model",registered_model_name = mlflow_config["registered_model_name"])
        else:
            mlflow.sklearn.load_model(lr,"model")
            







if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config",default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path = parsed_args.config)
