from typing import Optional
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from const import DATA_DIR

class Dataset:
    CATEGORICAL_COLUMNS = []
    NUMERICAL_COLUMNS = []
    TARGET_COLUMN = None

    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_train_test_split(self, test_size=0.2, random_state=42, data_sample: Optional[int] = None):
        data: pd.DataFrame = self.load_data()
        if data_sample:
            data = data.sample(n=data_sample, random_state=random_state)
        X = data[self.CATEGORICAL_COLUMNS + self.NUMERICAL_COLUMNS]
        y = data[self.TARGET_COLUMN]
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


class AustraliaDataset(Dataset):
    CATEGORICAL_COLUMNS = ["A1", "A4", "A5", "A6", "A8", "A9", "A11", "A12"]
    NUMERICAL_COLUMNS = ["A2", "A3", "A7", "A10", "A13", "A14"]
    TARGET_COLUMN = "A15"

    def load_data(self):
        file_path = os.path.join(self.data_path, "australian.dat")
        # Load the data using pandas
        data = pd.read_csv(file_path, delim_whitespace=True, header=None)
        # Define columns based on the dataset's structure
        data.columns = [f"A{i}" for i in range(1, 16)]
        return data


class GermanDataset(Dataset):
    CATEGORICAL_COLUMNS = ["A1", "A3", "A4", "A6", "A7", "A9", "A10", "A12", "A14", "A15", "A17", "A19", "A20" ]
    NUMERICAL_COLUMNS = ["A2", "A5", "A8", "A11", "A13", "A16", "A18"]
    TARGET_COLUMN = "A21"

    def load_data(self):
        file_path = os.path.join(self.data_path, "german.data")
        # Load the data using pandas
        data = pd.read_csv(file_path, delim_whitespace=True, header=None)
        data.columns = [f"A{i}" for i in range(1, 22)]
        data[self.TARGET_COLUMN] = data[self.TARGET_COLUMN].map(lambda x: 0 if int(x) == 2 else 1)
        return data


class CreditCardFraudDataset(Dataset):
    CATEGORICAL_COLUMNS = []  # No categorical columns except the target
    NUMERICAL_COLUMNS = [
        "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11",
        "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20", "V21", "V22",
        "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
    ]
    TARGET_COLUMN = "Class"

    def load_data(self):
        file_path = os.path.join(self.data_path, "creditcardfraud", "creditcard.csv")
        # Load the data using pandas
        data = pd.read_csv(file_path)
        # Define columns based on the dataset's structure
        data[self.TARGET_COLUMN] = data[self.TARGET_COLUMN].map(lambda x: int(x))
        return data


class LendingClubDataset(Dataset):
    CATEGORICAL_COLUMNS = [
        "term", "grade", "verification_status", "home_ownership",
        "initial_list_status", "application_type", "purpose", "addr_state"
    ]
    NUMERICAL_COLUMNS = [
        "loan_amnt", "int_rate", "installment", "annual_inc", "dti",
        "acc_now_delinq", "tax_liens", "delinq_amnt", "policy_code",
        "last_fico_range_high", "last_fico_range_low", "recoveries",
        "collection_recovery_fee"
    ]
    TARGET_COLUMN = "loan_status"

    def load_data(self):
        accepted_file_path = os.path.join(self.data_path, "lending-club", "accepted_2007_to_2018q4.csv", "accepted_2007_to_2018Q4.csv")
        
        # Load the accepted and rejected datasets
        data = pd.read_csv(accepted_file_path)
        
        data[self.TARGET_COLUMN] = data[self.TARGET_COLUMN].map(
            lambda x: 0 if x == "Fully Paid" else (
                1 if x in ["Late (16-30 days)", "Late (31-120 days)", "In Grace Period"] else None 
            )
        )
        data = data[data[self.TARGET_COLUMN].notna()]
        data = data[self.NUMERICAL_COLUMNS + self.CATEGORICAL_COLUMNS + [self.TARGET_COLUMN]]

        # Remove rows with NA values
        initial_row_count = len(data)
        data = data.dropna()
        filtered_row_count = len(data)
        
        # # # Print how many rows were filtered
        print(f"Initial row count: {initial_row_count}")
        print(f"Filtered row count: {filtered_row_count}")
        print(f"Filtered {initial_row_count - filtered_row_count} rows with NA values.")
        # Define columns based on the dataset's structure
        return data


def get_dataset_from_name(name) -> Dataset:
    if name == "australian":
        return AustraliaDataset(DATA_DIR)
    elif name == "german":
        return GermanDataset(DATA_DIR)
    elif name == "creditcardfraud":
        return CreditCardFraudDataset(DATA_DIR)
    elif name == "lendingclub":
        return LendingClubDataset(DATA_DIR)
    else:
        raise ValueError(f"Unknown dataset: {name}")