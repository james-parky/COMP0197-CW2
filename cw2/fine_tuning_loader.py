from typing import Any
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, Resize, ToTensor


class FineTuningLoader:
    """
    Class that builds and returns a torch DataLoader using the fine tuning dataset of
    cats and dogs. The dataset is a sample of images from both the Oxford-IIIT Pet
    dataset.
    """

    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers

    def build(self, ts: list[Any], split: float, path: str) -> DataLoader:
        """
        Apply any data transforms, and target transforms specified to the dataset,
        create a torch Loader and return it.

        Args:
            ts (list[Any]): A list of the transforms to be applied.
            split (float): The portion of the dataset to be sampled.
            path (str): The relative location on disk for the dataset to be
                        downloaded/found.

        Returns:
            (DataLoader): The data loader with any transforms applied.
        """

        target_transform = Compose([Resize((64, 64)), ToTensor()])

        dataset = OxfordIIITPet(
            root=path,
            split="trainval" if path == "finetune" else "test",
            target_types="segmentation",
            download=True,
            transform=ts,
            target_transform=target_transform,
        )
        finetune_dataset = random_split(
            dataset, [split * len(dataset), 1 - (split * len(dataset))]
        )

        return DataLoader(
            finetune_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
