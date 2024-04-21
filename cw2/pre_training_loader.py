"""
Module to create the new dataset.
"""
import os
import random
import shutil
from typing import Any

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import Compose
import random

from download_pre_training_data import EXTRACTED_FOLDER


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
        return img, -1

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

    os.makedirs(EXTRACTED_FOLDER, exist_ok=True)

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
                destination_path = os.path.join(EXTRACTED_FOLDER, image)
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


class PreTrainingLoader:
    """
    Class that builds and returns a torch DataLoader using the pre-training dataset of
    cats and dogs. The dataset is a selection of images from both the Stanford Dog, and
    Cat Breeds datasets, chosen such that the split of breeds is similar to Oxford-IIIT
    Pet.
    """

    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers

    def build(self, ts: list[Any]) -> DataLoader:
        """
        Apply any data transforms specified to the dataset, create a torch Loader and
        return it.

        Args:
            ts (list[Any]): A list of the transforms to be applied.

        Returns:
            (DataLoader): The data loader with any transforms applied.
        """

        dataset = ImageDataset(EXTRACTED_FOLDER, transform=ts)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        return dataloader
