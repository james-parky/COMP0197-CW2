"Module for downloading the new cat and dog datasets through gdown."
import os
import shutil
import gdown  # pylint: disable=import-error


TRAINING_DATASET_FILE_ID = "14B78jkv9L-6UJheRbHgCDyysq5v4GF3Q"
TRAINING_DATASET_PATH_ZIP = "./training_dataset.zip"
CAT_DATASET_PATH = "./training_dataset"


def download_file_from_google_drive(file_id, output_path):
    """
    Download a file from Google Drive given its file ID and save it to the specified
    output path.

    Args:
        file_id (str): The ID of the file to download from Google Drive.
        output_path (str): The path where the downloaded file will be saved.
    """
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
        print(f"Downloaded file: {output_path}")
    else:
        print(f"File already exists: {output_path}")


# def download_file_from_google_drive(file_id, output_path):
#     """
#     Download a file from Google Drive given its file ID and save it to the specified output path.
#
#     Args:
#         file_id (str): The ID of the file to download from Google Drive.
#         output_path (str): The path where the downloaded file will be saved.
#     """
#     url = f"https://drive.google.com/uc?id={file_id}&export=download"
#     session = requests.Session()
#     response = session.get(url, stream=True)
#
#     # Check if the response is an HTML page with a virus scan warning
#     if response.headers.get('Content-Type') == 'text/html':
#         # Extract the download form data from the HTML page
#         download_form_data = {}
#         for input_tag in re.findall(r'<input\s.*?>', response.text):
#             name = re.search(r'name="(.*?)"', input_tag).group(1)
#             value = re.search(r'value="(.*?)"', input_tag).group(1)
#             download_form_data[name] = value
#
#         # Submit the download form to confirm the download
#         download_url = 'https://drive.usercontent.google.com/download'
#         response = session.post(download_url, data=download_form_data, stream=True)
#
#     # Save the file to disk
#     with open(output_path, 'wb') as file:
#         shutil.copyfileobj(response.raw, file)
#
#     print(f"Downloaded file: {output_path}")
#     del response


if __name__ == "__main__":
    # Download training dataset
    os.makedirs(os.path.dirname(TRAINING_DATASET_PATH_ZIP), exist_ok=True)
    download_file_from_google_drive(
        TRAINING_DATASET_FILE_ID, TRAINING_DATASET_PATH_ZIP
    )

    # Extract the downloaded datasets if not already extracted
    if not os.path.exists(os.path.join(CAT_DATASET_PATH, "images")):
        shutil.unpack_archive(TRAINING_DATASET_PATH_ZIP, CAT_DATASET_PATH)
        print("Extracted training dataset.")
    else:
        print("Training dataset already extracted.")
