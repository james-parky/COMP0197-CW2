"""
Module to create the new dataset.
"""
import os
import random
import shutil
from typing import Tuple
from torch.utils.data import Dataset, DataLoader  # pylint: disable=import-error
from PIL import Image
from torchvision import transforms  # pylint: disable=import-error

# Specify the source folders and destination folder
SOURCE_FOLDER_1 = "./cats_filtered"
SOURCE_FOLDER_2 = "./dogs_filtered"
DESTINATION_FOLDER = "./training_dataset/training_dataset"


class ImageDataset(Dataset):
    """
    A custom PyTorch dataset for images.

    Args:
        root_dir (str): Root directory path containing the dataset.
        transform (callable, optional): Optional transform to be applied on a sample.
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = self._make_dataset()

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the image and a placeholder target (e.g., -1).
        """
        img_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, -1  # Return a placeholder target (e.g., -1)

    def _make_dataset(self):
        """
        Create the dataset by collecting image paths.

        Returns:
            list: List of image paths.
        """
        samples = []
        for img_name in os.listdir(self.root_dir):
            if img_name.endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(self.root_dir, img_name)
                samples.append(img_path)
        return samples

    # def download_file_from_google_drive(file_id, output_path):
    # """
    # Download a file from Google Drive given its file ID and save it to the specified output path.
    #
    # Args:
    #     file_id (str): The ID of the file to download from Google Drive.
    #     output_path (str): The path where the downloaded file will be saved.
    # """
    # if not os.path.exists(output_path):
    #     url = f"https://drive.google.com/uc?id={file_id}"
    #     gdown.download(url, output_path, quiet=False)
    #     print(f"Downloaded file: {output_path}")
    # else:
    #     print(f"File already exists: {output_path}")
    #
    # def download_file_from_google_drive(file_id, output_path):
    #     """
    #     Download a file from Google Drive given its file ID and save it to the specified
    #     output path.
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


def subsample_images(source_folder, num_images=200):
    """
    Randomly sub-sample images from a source folder containing multiple sub-folders,
    each containing multiple images. The sub-sampled images are then moved to a
    separate destination folder.

    Args:
        source_folder (str): Path to the source folder containing sub-folders with
                             images.
        DESTINATION_FOLDER (str): Path to the destination folder where sub-sampled
                                  images will be moved.
        num_images (int): Number of images to sub-sample from each sub-folder
                          (default is 200).
    """
    os.makedirs(DESTINATION_FOLDER, exist_ok=True)

    for subfolder in os.listdir(source_folder):
        subfolder_path = os.path.join(source_folder, subfolder)

        if os.path.isdir(subfolder_path):
            image_files = [
                f
                for f in os.listdir(subfolder_path)
                if f.endswith((".jpg", ".jpeg", ".png"))
            ]

            num_subsample = min(len(image_files), num_images)

            subsampled_images = random.sample(image_files, num_subsample)

            for image in subsampled_images:
                source_path = os.path.join(subfolder_path, image)
                destination_path = os.path.join(DESTINATION_FOLDER, image)
                shutil.copy(source_path, destination_path)


def count_files(folder_path):
    """
    Count the number of files within a specified folder.

    Args:
        folder_path (str): Path to the folder.

    Returns:
        int: Number of files in the folder.

    Raises:
        ValueError: If the specified path is not a valid directory.
    """
    if not os.path.isdir(folder_path):
        raise ValueError(f"{folder_path} is not a valid directory.")

    file_list = os.listdir(folder_path)
    file_count = len(file_list)

    return file_count


class UnsupervisedLoader:
    """
    Create a dataloader with the custom cat and dog dataset, with optional data data
    augmentation applied.
    """

    def __init__(
        self, img_size: Tuple[int, int] = (224, 224), batch_size: int = 32
    ):
        self.img_size = img_size
        self.batch_size = batch_size

    def build(self, data_augmentation=None):
        """
        Apply the optional data augmentation transforms if they exist and return the
        dataloader.
        """
        # Sub-sample images from the source folders
        # subsample_images(SOURCE_FOLDER_1)
        # subsample_images(SOURCE_FOLDER_2)

        # Define the data transformation

        base_transforms = [
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
            transforms.RandomGrayscale(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
        ]

        transform_composition = transforms.Compose(
            data_augmentation  # + base_transforms
            if data_augmentation
            else base_transforms
        )

        # Create the dataset and dataloader
        dataset = ImageDataset(
            DESTINATION_FOLDER, transform=transform_composition
        )
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        return dataloader
