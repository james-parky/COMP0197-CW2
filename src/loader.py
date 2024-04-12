"""
Module containing the implementation of the H5 Image Loader.
"""
import random
from typing import Iterator, Optional, Tuple
import numpy as np
import h5py
import torch

GRAYSCALE_VECTOR = [0.2989, 0.5870, 0.1140]


class H5ImageLoader:
    """
    Class to implement an image loader for .h5 image loaders. Images can be processed
    in full colour, or grayscale.
    """

    def __init__(
        self,
        img_file: str,
        batch_size: int,
        seg_file: Optional[str] = None,
        is_grayscale: bool = False,
    ) -> None:
        """
        Initialiser for the class.

        Args:
            img_file (str): The local path to the .h5 file containing the images for
                            which to create a dataloader.
            batch_size (int): The batch size of the dataloader.
            seg_file (Optional[str]): The local path to the .h5 files containing the
                                      corresponding labels for the images file.
            is_grayscale (bool): A boolean for whether the images should be converted
                                 to grayscale before the dataloader is created.
        """
        self.img_h5 = h5py.File(img_file, "r")
        self.dataset_list = list(self.img_h5.keys())
        if seg_file is not None:
            self.seg_h5 = h5py.File(seg_file, "r")
            if set(self.dataset_list) > set(self.seg_h5.keys()):
                raise IndexError("Images are not consistent with segmentation.")
        else:
            self.seg_h5 = None

        self.batch_size = batch_size
        self.num_batches = int(len(self.img_h5) / self.batch_size)
        self.img_ids = list(range(len(self.img_h5)))
        self.is_grayscale = is_grayscale
        self.batch_idx = 0

    def __len__(self):
        return len(self.dataset_list * self.batch_size)

    @staticmethod
    def rgb_to_grayscale(image: np.ndarray) -> np.ndarray:
        """
        Static method to convert a supplied image to grayscale using the matplotlib
        convert vector. Transformation only occurs if the image is RGB.

        Args:
            image (np.ndarray): The image to be converted.

        Returns:
            (np.ndarray): The converted image if the image was RGB, or the original
                          image.
        """

        if image.ndim == image.shape[-1] == 3:
            image = np.clip(np.dot(image, GRAYSCALE_VECTOR), 0, 255).astype(
                np.uint8
            )
        return image

    def __iter__(self) -> Iterator:
        """
        Iterator dunder method to shuffle the dataset and return an iterator.

        Returns:
            (Iterator): An iterator with shuffled data.
        """
        self.batch_idx = 0
        random.shuffle(self.img_ids)
        return self

    def __next__(self) -> Tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Next dunder to return the next batch of images and labels. If self.is_grayscale
        is set to True, the images will be converted before the batch is returned.

        Returns:
            (Tuple[list[np.ndarray], list[np.ndarray]]): The batch of images and labels.
        """
        self.batch_idx += 1
        batch_img_ids = self.img_ids[
            self.batch_idx
            * self.batch_size : (self.batch_idx + 1)
            * self.batch_size
        ]
        datasets = [self.dataset_list[idx] for idx in batch_img_ids]

        if self.batch_idx >= self.num_batches:
            raise StopIteration

        images = [self.img_h5[ds][()] for ds in datasets]
        labels = (
            None
            if (self.seg_h5 is None)
            else [self.seg_h5[ds][()] == 1 for ds in datasets]
        )  # foreground only
        images = [
            torch.from_numpy(self.img_h5[ds][()]).float() for ds in datasets
        ]
        labels = (
            None
            if self.seg_h5 is None
            else [torch.from_numpy(self.seg_h5[ds][()]) == 1 for ds in datasets]
        )
        # print(images, labels)
        return images, labels
