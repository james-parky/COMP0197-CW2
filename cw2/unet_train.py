""" 
Module for the fully supervised training of the UNet model.
"""
from unet_model import UNet
from typing import Any
from fine_tune import FineTuningLoader
from torch import nn
from label_segmentations import label_segmentations
import torch


def unet_train(
    model: UNet,
    device: str,
    ts: list[Any],
    num_epochs: int = 10,
    learning_rate: float = 0.06,
) -> None:
    """
    Train the UNet model using the pre-training dataset, with some optional transforms
    applied.

    Args:
        model (UNet): The UNet model to be trained.
        device (str): The current torch device string.
        ts (list[Any]): A list of the transforms to be applied.
        num_epochs (int): The number of epochs to train over.
        learning_rate (float): The learning rate of the optimiser.
    """

    model.to(device)
    model.train()

    dataloader = FineTuningLoader(batch_size=32, num_workers=4).build(
        ts, split=1.0, path="finetune"
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("Starting UNet Training...")
    for epoch in range(num_epochs):
        total_loss = 0.0

        for i, (images, masks) in enumerate(dataloader):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            masks = label_segmentations(masks)

            loss = criterion(outputs, masks)
            total_loss += loss.detach()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i % 40 == 39 or i in [0, len(dataloader) - 1]:
                print(
                    (
                        f"[Epoch {epoch + 1}/{num_epochs},"
                        f" Batch {i + 1}/{len(dataloader)}]:"
                        f" Loss = {loss.item():.5f}"
                    )
                )

        avg_loss = total_loss / len(dataloader)
        print(
            f"[Epoch: {epoch + 1}/{num_epochs}]: Average Loss = {avg_loss:.5f}"
        )

    print("Training complete.")
