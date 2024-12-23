import os
import yaml
import json
from datetime import datetime
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from datasets import get_dataset_from_name
import contextlib
import sys

from eval_utils import load_config, save_logs

def train_and_evaluate(config, log_dir):
    # Load dataset
    dataset_name = config['dataset']
    dataset = get_dataset_from_name(dataset_name)
    X_train, X_test, y_train, y_test = dataset.get_train_test_split(data_sample=config.get('data_sample', None))

    # Prepare data for CatBoost
    train_pool = Pool(
        X_train, y_train,
        cat_features=dataset.CATEGORICAL_COLUMNS,
        feature_names=X_train.columns.tolist(),
    )
    test_pool = Pool(
        X_test, y_test,
        cat_features=dataset.CATEGORICAL_COLUMNS,
        feature_names=X_train.columns.tolist(),
    )

    # Initialize and train the model
    log_file_path = os.path.join(log_dir, "catboost_training.log")
    with open(log_file_path, 'w') as log_file:
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            model = CatBoostClassifier(
                **config['catboost_params'],
                logging_level='Verbose',
                metric_period=10,
            )
            model.fit(train_pool, eval_set=test_pool, verbose=True)

    # Evaluate the model
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba)
    }

    return model, metrics, y_test, y_pred_proba


def main(config_path):
    config = load_config(config_path)
    log_dir = save_logs(config, {}, "catboost", config['dataset'], [], [])
    model, metrics, y_test, y_pred_proba = train_and_evaluate(config, log_dir)
    save_logs(config, metrics, "catboost", config['dataset'], y_test, y_pred_proba, log_dir)
    print(f"Training complete. Logs saved in {os.path.join('logs', config['dataset'])}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_catboost.py <config_path>")
    else:
        main(sys.argv[1])
