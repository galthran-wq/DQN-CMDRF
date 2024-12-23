import os
import yaml
import json
from datetime import datetime
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

import contextlib

import sys

from datasets import get_dataset_from_name
from eval_utils import load_config, save_logs



def train_and_evaluate(config):
    # Load dataset
    dataset_name = config['dataset']
    dataset = get_dataset_from_name(dataset_name)
    X_train, X_test, y_train, y_test = dataset.get_train_test_split()
    
    # Data preprocessing steps

    cat_transformer = make_column_transformer(
        (OrdinalEncoder(), dataset.CATEGORICAL_COLUMNS), 
        remainder='passthrough'
    )
    # Running grid search
    param_grid = {
        'rf__n_estimators': [50, 100, 150, 200],
        'rf__max_depth' : [None, 3, 5, 10, 20]
    }
    model = Pipeline([
        ('col_transformer', cat_transformer), 
        ('rf', RandomForestClassifier())]
    )
    gscv = GridSearchCV(
        model, param_grid=param_grid, 
        cv=5, scoring='f1', refit=True
    )
    gscv.fit(X_train, y_train)
    clf = gscv.best_estimator_
    
    # Evaluate the model
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba)
    }

    return clf, metrics, gscv.best_params_, y_test, y_pred_proba


def main(config_path):
    config = load_config(config_path)
    model, metrics, params, y_test, y_pred_proba = train_and_evaluate(config)
    save_logs(config, metrics, config['dataset'], y_test, y_pred_proba, gs_params=params)
    print(f"Training complete. Logs saved in {os.path.join('gridsearch_results', config['dataset'])}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python gscv_RF.py <config_path>")
    else:
        main(sys.argv[1])

  