"""
----------------
Module Imports
----------------
"""
import torch as tc
from torch import nn
import torchvision as tv
from oxford_data import OxfordData


class SimpleResnetModel:
    """
    ----------------
    Model Setup
    ----------------
    """

    def __init__(self):
        """
        Initialises the model parameters, calling all other init functions in the process.

        :return: 'None'; nothing is returned.
        """
        self.data = OxfordData()
        self.init_loader()
        self.init_device()
        self.init_net()

        self.net.to(self.device)

        self.criterion = tc.nn.CrossEntropyLoss()
        self.optimiser = tc.optim.Adam(
            self.net.parameters(), lr=0.001, weight_decay=0.01
        )

    def init_loader(self, size=20, work=8):
        """
        Initialises DataLoaders for training and testing purposes.

        :param size: The desired batch size to use, by default this is set as 20.
        :param work: The number of workers to use, by default this is set to be 8.
        """
        self.trainloader = tc.utils.data.DataLoader(
            self.data.get_train(),
            batch_size=size,
            shuffle=True,
            num_workers=work,
        )
        self.testloader = tc.utils.data.DataLoader(
            self.data.get_test(),
            batch_size=size,
            shuffle=True,
            num_workers=work,
        )

    def init_device(self):
        """
        Checks for CUDA availability and sets the 'self.device' instance variable to use it
        if available, otherwise defaulting to the CPU.

        :return: 'None'; nothing is returned.
        """
        if tc.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

    def init_net(self):
        """
        Initialises the network to be used as "self.net".

        :return: 'None'; nothing is returned.
        """
        self.net = tv.models.resnet18(weights=None)
        num_ftrs = self.net.fc.in_features
        self.net.fc = nn.Linear(num_ftrs, 37)

    def save_model(self):
        """
        Saves the current model.

        :return: 'None'; nothing is returned.
        """
        tc.save(self.net.state_dict(), "saved_simple_model.pt")
        print("Model saved")

    def load_model(self, name="saved_simple_model.pt"):
        """
        Defines a new model with loaded parameters.

        :param name: A string of the filename containing the desired model parameters.
        :return: 'None'; nothing is returned.
        """
        self.init_net()
        self.net.load_state_dict(tc.load(name))
        self.net.to(self.device)
        print("Model loaded")

    # ----------------
    # Model Training
    # ----------------

    def train(self, epochs=20):
        """
        Uses pre-initialised model parameters to train the network.

        :param epochs: An integer value representing number of epochs to train over.
        :return: 'None'; nothing is returned.
        """
        for epoch in range(epochs):
            print(f"\n[EPOCH {epoch+1}]")
            for data in self.trainloader:
                # print(f"Batch: {i}")
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimiser.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels.to(dtype=tc.long))
                loss.backward()
                self.optimiser.step()
            self.test()
        print("\n\nTraining complete")
        self.save_model()

    # ----------------
    # Model Testing
    # ----------------

    def test(self):
        """
        Runs tests on the trained model in batches.

        :return: 'None'; nothing is returned.
        """
        total = correct = 0
        print("\n[TEST]")
        for data in enumerate(self.testloader):
            # print(f"Batch: {i}")
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            output = self.net(inputs)
            _, predictions = tc.max(output, 1)
            # print("INPUT: ", inputs[0])
            # print("LABEL: ", labels[0])
            # print("PREDICTION: ", predictions[0])
            for i, prediction in enumerate(predictions):
                total += 1
                if prediction == labels[i]:
                    correct += 1
        accuracy = correct / total
        print("\nAccuracy Results:")
        print(f"Score: {correct} / {total}")
        print(f"Decimal: {accuracy}")


# ----------------
# Execution Script
# ----------------


def main():
    """
    Main path of execution for this model.
    Trains and tests a simple residual network model on the Oxford-IIIT Pet dataset.

    :return: 'None'; nothing is returned.
    """
    model = SimpleResnetModel()
    model.train()
    model.load_model()
    model.test()


if __name__ == "__main__":
    # Limits script to execute only when called directly.
    main()
