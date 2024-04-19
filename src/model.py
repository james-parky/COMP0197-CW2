"""
SimCLR and MaskedAutoencoder models and associated training/validation module.
"""
# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.
import torch  # pylint: disable=import-error
import torchvision  # pylint: disable=import-error
from torch import nn  # pylint: disable=import-error
from lightly.loss import NTXentLoss  # pylint: disable=import-error
from lightly.models.modules import (  # pylint: disable=import-error
    SimCLRProjectionHead,  # pylint: disable=import-error
)  # pylint: disable=import-error
from lightly.transforms.simclr_transform import (  # pylint: disable=import-error
    SimCLRTransform,  # pylint: disable=import-error
)  # pylint: disable=import-error

# from sklearn.metrics import (
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
# )
from unsupervised_loader import UnsupervisedLoader


class SimCLR(nn.Module):
    """
    SimCLR self-supervised model implementation.
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SimCLRProjectionHead(512, 512, 128)

    def forward(self, x):
        """
        Forward pass.
        """
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z


def train_simclr(model, num_epochs: int = 10):
    """
    Training method for the SimCLR model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    # transform = SimCLRTransform(input_size=32, gaussian_blur=0.0)
    #
    #
    # dataset = torchvision.datasets.CIFAR10(
    #     "datasets/cifar10", download=True, transform=transform
    # )

    # or create a dataset from a folder containing images or videos:
    # dataset = LightlyDataset("path/to/folder", transform=transform)

    dataloader = UnsupervisedLoader(img_size=(64, 64), batch_size=32).build(
        data_augmentation=[SimCLRTransform(input_size=32, gaussian_blur=0.0)]
    )

    criterion = NTXentLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.06)

    print("Starting Training")
    for epoch in range(num_epochs):
        total_loss = 0
        for i, batch in enumerate(dataloader):
            x0, x1 = batch[0]
            x0 = x0.to(device)
            x1 = x1.to(device)

            z0 = model(x0)
            z1 = model(x1)

            loss = criterion(z0, z1)
            total_loss += loss.detach()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f"[Epoch {epoch + 1}, Batch {i + 1}]: {loss.item()}")

        avg_loss = total_loss / len(dataloader)
        print(f"epoch: {epoch + 1}, loss: {avg_loss:.5f}")

    # Reporting section
    print("Training completed.")

    # # Prepare test dataset
    # test_transform = SimCLRTransform(input_size=32, gaussian_blur=0.0)
    # test_dataset = torchvision.datasets.CIFAR10(
    #     "datasets/cifar10", download=True, transform=test_transform, train=False
    # )
    # test_dataloader = torch.utils.data.DataLoader(
    #     test_dataset, batch_size=256, shuffle=False, num_workers=8
    # )
    #
    #
    # # Evaluate the model on the test dataset
    # model.eval()
    # y_true = []
    # y_pred = []
    # with torch.no_grad():
    #     for batch in test_dataloader:
    #         x0, x1 = batch[0]
    #         x0 = x0.to(device)
    #         x1 = x1.to(device)
    #
    #         z0 = model(x0)
    #         z1 = model(x1)
    #
    #         # Compute cosine similarity between z0 and z1
    #         similarity = torch.cosine_similarity(z0, z1, dim=1)
    #
    #         # Threshold the similarity scores to obtain predicted labels
    #         y_pred.extend((similarity > 0.5).cpu().numpy())
    #
    #         # Collect true labels
    #         y_true.extend(batch[1].numpy())

    # Compute evaluation metrics
    # accuracy = accuracy_score(y_true, y_pred)
    # precision = precision_score(y_true, y_pred, average='macro')
    # recall = recall_score(y_true, y_pred, average='macro')
    # f1 = f1_score(y_true, y_pred, average='macro')
    #
    #
    # # Print the evaluation metrics
    # print(f"Accuracy: {accuracy:.4f}")
    # print(f"Precision: {precision:.4f}")
    # print(f"Recall: {recall:.4f}")
    # print(f"F1 Score: {f1:.4f}")


class MaskedAutoencoder:
    """
    Class implementation of a masked autoencoder self-supervised model.
    """


def get_model(model: str = "autoencoder"):
    """
    Return a model and accompanying training function.
    """
    return (
        (MaskedAutoencoder(), None)
        if model == "autoencoder"
        else (
            SimCLR(
                nn.Sequential(
                    *list(torchvision.models.resnet18().children())[:-1]
                )
            ),
            train_simclr,
        )
    )


simclr_model, train = get_model("simclr")

if train is not None:
    train(simclr_model, num_epochs=1)


# resnet = torchvision.models.resnet18()
# backbone = nn.Sequential(*list(resnet.children())[:-1])
# model = SimCLR(backbone)
#
#
