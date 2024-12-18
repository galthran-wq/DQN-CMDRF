import yaml
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


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
