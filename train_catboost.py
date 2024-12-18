import os
import yaml
import json
from datetime import datetime
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt
from datasets import get_dataset_from_name
import contextlib
import sys

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def train_and_evaluate(config, log_dir):
    # Load dataset
    dataset_name = config['dataset']
    dataset = get_dataset_from_name(dataset_name)
    X_train, X_test, y_train, y_test = dataset.get_train_test_split()

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

def plot_and_save_curves(y_test, y_pred_proba, log_dir):
    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.figure()
    plt.plot(recall, precision, marker='.', label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    pr_curve_path = os.path.join(log_dir, 'precision_recall_curve.png')
    plt.savefig(pr_curve_path)
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure()
    plt.plot(fpr, tpr, marker='.', label='ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    roc_curve_path = os.path.join(log_dir, 'roc_curve.png')
    plt.savefig(roc_curve_path)
    plt.close()

def save_logs(config, metrics, dataset_name, y_test, y_pred_proba, log_dir=None):
    if log_dir is None:
        # Create log directory
        time_str = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
        log_dir = os.path.join("logs", dataset_name, time_str)
        os.makedirs(log_dir, exist_ok=True)

    # Save config
    config_path = os.path.join(log_dir, "config.yaml")
    with open(config_path, 'w') as file:
        yaml.dump(config, file)

    if len(y_test) > 0 and len(y_pred_proba) > 0:
        # Save metrics
        metrics_path = os.path.join(log_dir, "metrics.json")
        with open(metrics_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        # Save plots
        plot_and_save_curves(y_test, y_pred_proba, log_dir)

    return log_dir

def main(config_path):
    config = load_config(config_path)
    log_dir = save_logs(config, {}, config['dataset'], [], [])
    model, metrics, y_test, y_pred_proba = train_and_evaluate(config, log_dir)
    save_logs(config, metrics, config['dataset'], y_test, y_pred_proba, log_dir)
    print(f"Training complete. Logs saved in {os.path.join('logs', config['dataset'])}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_catboost.py <config_path>")
    else:
        main(sys.argv[1])
