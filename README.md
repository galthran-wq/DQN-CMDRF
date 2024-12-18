# Deep Q-Network with Confusion-Matrix-Based Dynamic Reward Function (DQN-CMDRF)

This repository implements the **DQN-CMDRF model**, a Deep Q-Network enhanced with a confusion-matrix-based dynamic reward function, as proposed in the paper *"Deep reinforcement learning with the confusion-matrix-based dynamic reward function for customer credit scoring"*. The project also includes comparisons with baseline methods to evaluate its performance on binary classification datasets.

---

## Features
- Implementation of the **DQN-CMDRF model** using PyTorch.
- Support for **categorical and numerical features**.
- Integration of a **confusion-matrix-based dynamic reward function**.
- Comparison with baseline models, including:
  - Artificial Neural Network (ANN)
  - Logistic Regression (LR)
  - Decision Tree (DT)
  - Random Forest (RF)
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
- Training logs, evaluation metrics, and plots of performance curves (Precision-Recall and ROC).

---

## Project Structure

```
.
├── train_dqn.py              # Main script for training the DQN-CMDRF model
├── datasets.py               # Dataset loading and preprocessing utilities
├── config.yaml               # Configuration file for hyperparameters and dataset settings
├── readme_dqn_cmdrf.md       # This README document
├── logs/                     # Directory for storing logs, models, and evaluation plots
└── requirements.txt          # Required Python packages
```

---

## Requirements

- Python >= 3.8
- PyTorch >= 1.10
- Scikit-learn
- Matplotlib
- Pandas

Install the dependencies using:

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Download data

#### Downloading and Extracting Data

The `download_data.py` script is used to download and extract datasets. It supports downloading from URLs and Kaggle datasets.

##### Steps to Download and Extract Data

1. **Run the Script**: Use the following command to download and extract datasets:

    ```bash
    python download_data.py main
    ```

2. **Functionality**:
   - The script defines a dictionary `DATASETS` with dataset information, including the path and type (either "url" or "kaggle").
   - It iterates over this dictionary and calls the `download_and_extract` function for each dataset.
   - The `download_and_extract` function downloads the dataset and extracts it if it's a zip file.

3. **Extracting Existing Zip Files**:
   - If you have zip files in the `data` directory, you can use the `extract_data` function to extract them.
   - This function iterates over files in the `data` directory, checks for zip files, and extracts them to a directory with the same name as the zip file (without the extension).

##### Example Datasets

The script is configured to download the following datasets:

- **Australia**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/australian/australian.dat)
- **German**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data)
- **Credit Card Fraud**: Kaggle dataset `mlg-ulb/creditcardfraud`
- **Lending Club**: Kaggle dataset `wordsforthewise/lending-club`

##### Resulting file structure

```
data/
├── australian.dat
├── creditcardfraud
│   └── creditcard.csv
├── creditcardfraud.zip
├── german.data
├── lending-club
│   ├── accepted_2007_to_2018q4.csv
│   │   └── accepted_2007_to_2018Q4.csv
│   ├── accepted_2007_to_2018Q4.csv.gz
│   ├── rejected_2007_to_2018q4.csv
│   │   └── rejected_2007_to_2018Q4.csv
│   └── rejected_2007_to_2018Q4.csv.gz
└── lending-club.zip
```

### 1. Configure Settings
Edit the `config.yaml` file to specify:
- The dataset name (e.g., "german" or "australia").
- Training parameters such as the number of episodes, batch size, and learning rate.

Example configuration:
```yaml
dataset: german
algorithm_params:
  episodes: 500
  batch_size: 64
  learning_rate: 0.001
  gamma: 0.99
  epsilon_start: 1.0
  epsilon_min: 0.01
  epsilon_decay: 0.995
```

### 2. Train the Model
Run the training script with the specified configuration file:

```bash
python train_dqn.py config.yaml
```

Logs and models will be saved in the `logs/` directory, organized by dataset and timestamp.

#### Log Folder Structure

After training, the logs and results are saved in a structured directory under `logs/<dataset_name>/<timestamp>/`. The structure is as follows:

```
logs/
└── <dataset_name>/
    └── <timestamp>/
        ├── config.yaml
        ├── metrics.json
        ├── precision_recall_curve.png
        └── roc_curve.png
```

- `config.yaml`: Contains the configuration used for the training.
- `metrics.json`: Stores the evaluation metrics such as accuracy, F1 score, precision, recall, and AUC.
- `precision_recall_curve.png`: A plot of the precision-recall curve.
- `roc_curve.png`: A plot of the ROC curve.


### 3. Evaluate the Model
After training, the script evaluates the model on the test dataset and computes metrics, including:
- Accuracy
- F1 Score
- Precision
- Recall
- AUC (Area Under ROC Curve)

Plots of the **Precision-Recall Curve** and **ROC Curve** are also saved in the `logs/` directory.

---

## Comparison with Baseline Methods

The project includes comparisons of the DQN-CMDRF model against the following baseline models:
- Artificial Neural Network (ANN)
- Logistic Regression (LR)
- Decision Tree (DT)
- Random Forest (RF)
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

Performance is evaluated on multiple binary classification datasets, with metrics such as **AUC**, **F1 Score**, and **Accuracy**.

---

## Results

| Model         | Dataset    | TODO      | TODO      | TODO   | TODO      | TODO   |
|---------------|------------|-----------|-----------|--------|-----------|--------|
| DQN-CMDRF     | German     | TODO      | TODO      | TODO   | TODO      | TODO   |
| ANN           | German     | TODO      | TODO      | TODO   | TODO      | TODO   |
| Logistic Reg. | German     | TODO      | TODO      | TODO   | TODO      | TODO   |
| Random Forest | German     | TODO      | TODO      | TODO   | TODO      | TODO   |

The **DQN-CMDRF model** demonstrates superior performance in terms of accuracy, F1 Score, and AUC compared to traditional classification models.


## References
- Wang, Y., Jia, Y., Tian, Y., & Xiao, J. (2022). *Deep reinforcement learning with the confusion-matrix-based dynamic reward function for customer credit scoring*. Expert Systems with Applications, 200, 117013.


