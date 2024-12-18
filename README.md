# Data Download and Extraction

This guide explains how to download and extract datasets using the provided Python scripts.

## Prerequisites

Ensure you have Python installed along with the necessary packages. You can install the required packages using the following command:

```bash
pip install -r requirements.txt
```

## Downloading and Extracting Data

The `download_data.py` script is used to download and extract datasets. It supports downloading from URLs and Kaggle datasets.

### Steps to Download and Extract Data

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

### Example Datasets

The script is configured to download the following datasets:

- **Australia**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/australian/australian.dat)
- **German**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data)
- **Credit Card Fraud**: Kaggle dataset `mlg-ulb/creditcardfraud`
- **Lending Club**: Kaggle dataset `wordsforthewise/lending-club`

### Resulting file structure

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

### Notes

- Ensure you have access to Kaggle datasets by setting up your Kaggle API credentials if you are downloading from Kaggle.
- The script uses the `fire` library to create a command-line interface.

By following these instructions, you can easily download and extract datasets for your project.
