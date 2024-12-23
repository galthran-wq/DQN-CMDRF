import yaml
import json
import os
from datetime import datetime
import numpy as np
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

    # Save fpr and tpr
    rates = np.empty((len(fpr), 2))
    rates[:, 0] = fpr
    rates[:, 1] = tpr
    rates = rates.tolist()
    rates_path = os.path.join(log_dir, "rates.json")
    with open(rates_path, 'w') as file:
        json.dump(rates, file, indent=4)



def save_logs(config, metrics, dataset_name, y_test, y_pred_proba, log_dir=None, gs_params=None, history=None):
    if log_dir is None:
        # Create log directory
        time_str = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
        log_dir = os.path.join("logs", dataset_name, time_str)
        if gs_params is not None:
            log_dir = os.path.join("gridsearch", dataset_name, time_str)
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

    # Save gridsearch parameters
    if gs_params is not None:
        gs_params_path = os.path.join(log_dir, "gs_params.json")
        with open(gs_params_path, 'w') as file:
            json.dump(gs_params, file, indent=4)
    
    # Save convergence plot
    if history is not None:
        plot_history(history, log_dir)
    
    return log_dir

    
def plot_history(history, log_dir):
    epoch_num = len(history)
    names_list = history[0].keys()
    metrics_arr = [[metrics[name] for metrics in history] for name in names_list]
    metrics_arr = np.array(metrics_arr)
    plt.figure()
    for i, name in enumerate(names_list):
        if (name == 'precision') or (name == 'recall'):
            continue
        plt.plot(np.arange(1, 1+epoch_num), metrics_arr[i, :], label=name)
    plt.ylim((0.5, 1))
    plt.xlabel('Epoch')
    plt.ylabel('Metric value')
    plt.title('Convergence plot')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(log_dir, 'convergence.png')
    plt.savefig(plot_path)
    plt.close()
    return
