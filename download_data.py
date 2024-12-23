import os
import requests
import zipfile
import kagglehub
import fire
import shutil


def download_and_extract(type, url, output_path, extract_path=None):
    """Download and extract a dataset from a URL"""
    os.makedirs(output_path, exist_ok=True)
    filename = os.path.join(output_path, url.split('/')[-1])

    if not os.path.exists(filename):
        if type == "url":
            print(f"Downloading {url}...")
            response = requests.get(url)
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"Saved to {filename}")
        elif type == "kaggle":
            print(f"Downloading {url}...")
            path = kagglehub.dataset_download(url, output_path)
            shutil.move(path, filename)
            print(f"Saved to {filename}")
        else:
            raise ValueError(f"Invalid download type: {type}")
    else:
        print(f"File already exists: {filename}")

    if extract_path and filename.endswith('.zip'):
        print("Extracting files...")
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("Extraction complete.")


data_dir = "data"


def main():
    # Dataset URLs
    DATASETS = {
        "australia": {
            "path": "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/australian/australian.dat",
            "type": "url"
        },
        "german": {
            "path": "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data",
            "type": "url"
        },
        # might not work
        # if so, use:
        # kaggle datasets download mlg-ulb/creditcardfraud
        # kaggle datasets download wordsforthewise/lending-club
        "creditcardfraud": {
            "path": "mlg-ulb/creditcardfraud",
            "type": "kaggle"
        },
        "lending_club": {
            "path": "wordsforthewise/lending-club",
            "type": "kaggle"
        }
    }

    # Download publicly available datasets
    for download_info in DATASETS.values():
        download_and_extract(download_info["type"], download_info["path"], data_dir)

    print("Download completed.")


def extract_data():
    for file in os.listdir(data_dir):
        if file.endswith(".zip"):
            name = file[:-4]  # Remove the .zip extension
            extract_path = os.path.join(data_dir, name)
            os.makedirs(extract_path, exist_ok=True)
            with zipfile.ZipFile(os.path.join(data_dir, file), 'r') as zip_ref:
                zip_ref.extractall(extract_path)


if __name__ == '__main__':
    fire.Fire()
    main()
