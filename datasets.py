import pandas as pd
from sklearn.model_selection import train_test_split
import os

from const import DATA_DIR

class Dataset:
    CATEGORICAL_COLUMNS = []
    NUMERICAL_COLUMNS = []
    TARGET_COLUMN = None

    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_train_test_split(self, test_size=0.2, random_state=42):
        data = self.load_data()
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
        data = pd.read_csv(file_path, sep=' ', header=None)
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
        data = pd.read_csv(file_path, sep=' ', header=None)
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
        "term", "grade", "sub_grade", "emp_title", "emp_length", "home_ownership",
        "verification_status", "pymnt_plan", "purpose", "title",
        "zip_code", "addr_state", "initial_list_status", "application_type",
        "hardship_flag", "hardship_type", "hardship_reason", "hardship_status",
        "hardship_loan_status", "disbursement_method", "debt_settlement_flag",
        "settlement_status"
    ]
    NUMERICAL_COLUMNS = [
        "loan_amnt", "funded_amnt", "funded_amnt_inv", "int_rate", "installment",
        "annual_inc", "dti", "delinq_2yrs", "fico_range_low", "fico_range_high",
        "inq_last_6mths", "mths_since_last_delinq", "mths_since_last_record",
        "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc", "out_prncp",
        "out_prncp_inv", "total_pymnt", "total_pymnt_inv", "total_rec_prncp",
        "total_rec_int", "total_rec_late_fee", "recoveries", "collection_recovery_fee",
        "last_pymnt_amnt", "collections_12_mths_ex_med", "mths_since_last_major_derog",
        "policy_code", "acc_now_delinq", "tot_coll_amt", "tot_cur_bal", "open_acc_6m",
        "open_act_il", "open_il_12m", "open_il_24m", "mths_since_rcnt_il",
        "total_bal_il", "il_util", "open_rv_12m", "open_rv_24m", "max_bal_bc",
        "all_util", "total_rev_hi_lim", "inq_fi", "total_cu_tl", "inq_last_12m",
        "acc_open_past_24mths", "avg_cur_bal", "bc_open_to_buy", "bc_util",
        "chargeoff_within_12_mths", "delinq_amnt", "mo_sin_old_il_acct",
        "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl", "mort_acc",
        "mths_since_recent_bc", "mths_since_recent_bc_dlq", "mths_since_recent_inq",
        "mths_since_recent_revol_delinq", "num_accts_ever_120_pd", "num_actv_bc_tl",
        "num_actv_rev_tl", "num_bc_sats", "num_bc_tl", "num_il_tl", "num_op_rev_tl",
        "num_rev_accts", "num_rev_tl_bal_gt_0", "num_sats", "num_tl_120dpd_2m",
        "num_tl_30dpd", "num_tl_90g_dpd_24m", "num_tl_op_past_12m", "pct_tl_nvr_dlq",
        "percent_bc_gt_75", "pub_rec_bankruptcies", "tax_liens", "tot_hi_cred_lim",
        "total_bal_ex_mort", "total_bc_limit", "total_il_high_credit_limit"
    ]
    TARGET_COLUMN = "loan_status"

    def load_data(self):
        accepted_file_path = os.path.join(self.data_path, "lending-club", "accepted_2007_to_2018q4.csv", "accepted_2007_to_2018Q4.csv")
        rejected_file_path = os.path.join(self.data_path, "lending-club", "rejected_2007_to_2018q4.csv", "rejected_2007_to_2018Q4.csv")
        
        # Load the accepted and rejected datasets
        accepted_data = pd.read_csv(accepted_file_path)
        rejected_data = pd.read_csv(rejected_file_path)
        
        # Concatenate the datasets
        data = pd.concat([accepted_data, rejected_data], ignore_index=True)

        data[self.TARGET_COLUMN] = data[self.TARGET_COLUMN].map(lambda x: 1 if x == "Fully Paid" else 0)
        
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