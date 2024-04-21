""" 
Module for the pre-training of the ViTMAE model.
"""

import torch
from pre_training_loader import PreTrainingLoader
from typing import Any
from transformers import ViTMAEForPreTraining

PRETRAINED_MODEL_PATH = "./pretrained_model.pth"


def vit_mae_train(
    model: ViTMAEForPreTraining,
    device: str,
    ts: list[Any],
    num_epochs: int = 10,
    learning_rate: float = 0.06,
):
    """
    Train the ViTMAE model using the pre-training dataset, with some optional transforms
    applied.

    Args:
        model (ViTMAEForPreTraining): The ViTMAE model to be trained.
        device (str): The current torch device string.
        ts (list[Any]): A list of the transforms to be applied.
        num_epochs (int): The number of epochs to train over.
        learning_rate (float): The learning rate of the optimiser.
    """
    model.to(device)
    model.train()

    dataloader = PreTrainingLoader(batch_size=32, num_workers=4).build(ts)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("Starting ViTMAE Training...")
    for epoch in range(num_epochs):
        total_loss = 0.0

        for i, batch in enumerate(dataloader):
            images, _ = batch  # Ignore the labels during pretraining
            images = images.to(device)
            outputs = model(pixel_values=images)

            loss = outputs.loss
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()

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

    torch.save(model.state_dict(), PRETRAINED_MODEL_PATH)
    print("Pretraining completed. Model saved.")
