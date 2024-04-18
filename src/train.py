"""
Module to train an image segmenation, based on Yipengs tutorial code.
"""
from typing import Tuple
import os
import torch
import albumentations as A
from net import ResUNet
from loader import H5ImageLoader
from download_data import DATA_PATH

MINIBATCH_SIZE = 32
NETWORK_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 5
FREQ_INFO = 1
FREQ_SAVE = 100
SAVE_PATH = "results-pt"


transform = A.Compose(
    [
        A.ChromaticAberration(
            mode="red_blue",
            primary_distortion_limit=0.5,
            secondary_distortion_limit=0.1,
            p=1,
        ),
        A.RandomCrop(width=50, height=50),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ]
)


def dice_score(ps: torch.Tensor, ts: torch.Tensor, eps: float = 1e-6) -> float:
    """
    Calculate the dice score for the given tensors, in other words, their similarity
    coefficient.

    Args:
        ps (torch.Tensor): The predicted values.
        ts (torch.Tensor): The target values.
        eps (float): A small epsilon value to avoid zeros, defaulted to 1e-6.

    Returns:
        (float): The dice score.
    """
    numerator = torch.sum(ts * ps, dim=[1, 2, 3]) * 2 + eps
    denominator = (
        torch.sum(ts, dim=[1, 2, 3]) + torch.sum(ps, dim=[1, 2, 3]) + eps
    )
    return numerator / denominator


def pre_process(
    images: list[torch.Tensor], labels: list[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Preprocess the data the get it into the correct format for the image segmentation
    model. The images need to be permutated to ensure the shape is correct.

    Args:
        images (list[torch.Tensor]): A batch of 32 (64x64) full RGB images.
        labels (list[torch.Tensor]): The associated boolean tensor indicating which
        pixels are part of the foreground segmentation.

    Returns:
        (Tuple[torch.Tensor, torch.Tensor]): The images and labels in the correct format
        for the model.
    """
    images = torch.stack(images)
    images = images.permute(0, 3, 1, 2)
    labels = torch.stack(labels).unsqueeze(1)
    return images.float(), labels.float()


def train_model(loader: H5ImageLoader) -> None:
    """
    Train the image segmentation model on the provided H5ImageLoader, made from the
    Oxford pet dataset.

    Args:
        loader_train (H5ImageLoader): The image loader containing images for training.
    """
    seg_net = ResUNet(init_ch=NETWORK_SIZE)
    optimiser = torch.optim.Adam(seg_net.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for images, labels in loader:
            images, labels = pre_process(images, labels)
            predicts = seg_net(images)
            loss = torch.mean(1 - dice_score(predicts, labels))
            optimiser.zero_grad()
            loss.backward()
            running_loss += loss.item()
            optimiser.step()

        if epoch + 1 % FREQ_INFO == 0:
            print(f"[epoch {epoch+1}]: loss={running_loss/FREQ_INFO}")
            running_loss = 0.0


if __name__ == "__main__":
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    loader_train = H5ImageLoader(
        f"{DATA_PATH}/images_train.h5",
        MINIBATCH_SIZE,
        f"{DATA_PATH}/labels_train.h5",
        transform=transform,
    )
    train_model(loader_train)
