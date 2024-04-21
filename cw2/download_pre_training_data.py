"""
Module for downloading the new cat and dog datasets through gdown.
"""
import os
import shutil
import gdown


TRAINING_DATASET_FILE_ID = "14B78jkv9L-6UJheRbHgCDyysq5v4GF3Q"
TRAINING_DATASET_PATH_ZIP = "./training_dataset.zip"
DATASET_PATH = "./training_dataset"
SOURCE_FOLDER_1 = "./cats_filtered"
SOURCE_FOLDER_2 = "./dogs_filtered"
EXTRACTED_FOLDER = "./training_dataset/training_dataset"


def download_file_from_google_drive(file_id: str, output_path: str) -> None:
    """
    Download a file from Google Drive given its file ID and save it to the specified
    output path.

    Args:
        file_id (str): The ID of the file to download from Google Drive.
        output_path (str): The path where the downloaded file will be saved.
    """
    os.makedirs(os.path.dirname(TRAINING_DATASET_PATH_ZIP), exist_ok=True)

    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
        print(f"Downloaded file: {output_path}")
    else:
        print(f"File already exists: {output_path}")


if __name__ == "__main__":
    download_file_from_google_drive(
        TRAINING_DATASET_FILE_ID, TRAINING_DATASET_PATH_ZIP
    )

    if not os.path.exists(os.path.join(DATASET_PATH, "images")):
        shutil.unpack_archive(TRAINING_DATASET_PATH_ZIP, DATASET_PATH)
        print("Extracted training dataset.")
    else:
        print("Training dataset already extracted.")
