"""
----------------
Module Imports
----------------
"""
import os
import ssl
import torch as tc
import torchvision


class OxfordData:
    """
    OxfordData class for downloading the Oxford-IIIT Pet dataset and storing it for use.
    """

    def __init__(self, root="../data/oxford_pets", transform=None):
        """
        Initialises OxfordData object.

        :param root: String representing the root data folder, by default this
                     is '/data' in top level of '/src' (when called from the 'supervised' folder).
        :param transform: torchvision.transforms object defining transforms to
                          be performed on data upon download, by default this is 'None'.
        :return: 'None'; nothing is returned.
        """
        # Prevent certificate verification error when downloading Torchvision datasets
        ssl._create_default_https_context = ssl._create_unverified_context

        self.root = root.replace("/", os.path.sep)
        self.trainroot = os.path.join(self.root, "train")
        self.testroot = os.path.join(self.root, "test")
        if transform:
            self.transform = transform
        else:
            # Default transformations
            self.transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize((224, 224)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                    ),
                ]
            )
        self.ingest_data()

    def ingest_data(self):
        """
        Downloads the Oxford-IIIT Pet dataset into the object's pre-defined root folder.

        :return: 'None'; nothing is returned.
        """
        self.train = torchvision.datasets.OxfordIIITPet(
            root=self.trainroot,
            split="trainval",
            transform=self.transform,
            download=True,
        )
        self.test = torchvision.datasets.OxfordIIITPet(
            root=self.testroot,
            split="test",
            transform=self.transform,
            download=True,
        )

    def get_train(self):
        """
        Returns the training data.

        :return: A TorchVision dataset stored in the 'self.train' instance variable or 'None'.
        """
        if self.train:
            return self.train
        raise ValueError("OxfordData: No training data to return.")

    def get_test(self):
        """
        Returns the training data.

        :return: A TorchVision dataset stored in the 'self.train' instance variable or 'None'.
        """
        if self.test:
            return self.test
        raise ValueError("OxfordData: No test data to return.")

    def get_subset(self, subset, size):
        """
        Returns a subset of the specified dataset of the specified size.

        :param set: A string representing the choice of dataset,
                    "test" being for self.test and self.train being the default.
        :param size: An integer representing the size of the desired subset.
        :return: A TorchVision dataset that is a subset of either 'self.test'
                 or 'self.train' instance variables.
        """
        if subset == "test":
            data = self.test
        else:
            data = self.train
        indices = list(range(size))

        return tc.utils.data.Subset(data, indices)
