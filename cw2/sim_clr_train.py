""" 
Module for the pre-training of the SimCLR model.
"""
import torch
from sim_clr_model import SimCLR
from pre_training_loader import PreTrainingLoader
from lightly.loss import NTXentLoss
from typing import Any


def sim_clr_train(
    model: SimCLR,
    device: str,
    ts: list[Any],
    num_epochs: int = 10,
    learning_rate: float = 0.06,
):
    """
    Train the SimCLR model using the pre-training dataset, with some optional transforms
    applied.

    Args:
        model (SimCLR): The SimCLR model to be trained.
        device (str): The current torch device string.
        ts (list[Any]): A list of the transforms to be applied.
        num_epochs (int): The number of epochs to train over.
        learning_rate (float): The learning rate of the optimiser.
    """

    model.to(device)
    model.train()

    dataloader = PreTrainingLoader(batch_size=32, num_workers=4).build(ts)
    criterion = NTXentLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("Starting SimCLR Training...")
    for epoch in range(num_epochs):
        total_loss = 0.0

        for i, batch in enumerate(dataloader):
            x0, x1 = batch[0]
            x0 = x0.to(device)
            x1 = x1.to(device)
            z0 = model(x0, return_embeddings=True)
            z1 = model(x1, return_embeddings=True)

            loss = criterion(z0, z1)
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
