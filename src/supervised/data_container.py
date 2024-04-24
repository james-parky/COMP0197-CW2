"""
----------------
Module Imports
----------------
"""
import os
import ssl
import torch as tc
import torchvision


class DataContainer:
    """
    DataContainer class for either CIFAR-10 or Oxford-IIIT Pet datasets.
    This class downloads and stores them, providing some utility functions for use.
    """

    def __init__(self, root="../data", dataset="oxford", transform=None):
        """
        Initialises DataContainer object.

        :param root: String representing the root data folder, by default this
                     is '/data' in top level of '/src' (when called from the 'supervised' folder).
        :param dataset: String representing which of the datasets to use, by default this is
                        'oxford', but can also take the value 'cifar'.
        :param transform: torchvision.transforms object defining transforms to
                          be performed on data upon download, by default this is 'None'.
        :return: 'None'; nothing is returned.
        """
        # Prevent certificate verification error when downloading Torchvision datasets
        ssl._create_default_https_context = ssl._create_unverified_context

        self.dataset = dataset
        self.root = root + "/" + dataset
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
        Downloads the dataset specified by 'self.dataset' into the object's
        pre-defined root folder.

        :return: 'None'; nothing is returned.
        """
        if self.dataset == "cifar":
            self.train = torchvision.datasets.CIFAR10(
                root=self.trainroot,
                train=True,
                transform=self.transform,
                download=True,
            )
            self.test = torchvision.datasets.CIFAR10(
                root=self.testroot,
                train=False,
                transform=self.transform,
                download=True,
            )
        else:
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

        print(f"Successfully Downloaded: {self.dataset}")

    def get_train(self):
        """
        Returns the training data.

        :return: A TorchVision dataset stored in the 'self.train' instance variable or 'None'.
        """
        if self.train:
            return self.train
        raise ValueError("DatasetContainer: No training data to return.")

    def get_test(self):
        """
        Returns the training data.

        :return: A TorchVision dataset stored in the 'self.train' instance variable or 'None'.
        """
        if self.test:
            return self.test
        raise ValueError("DatasetContainer: No test data to return.")

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
