from fine_tuning_loader import FineTuningLoader
from torch import nn
import torch
from label_segmentations import label_segmentations


def fine_tune(model, device, ts, num_epochs, learning_rate, split) -> None:
    """
    Fine tune the given model on a sample of size split, of the Oxford-IIIT Pet dataset.
    Print periodic and final average loss.

    Args:
        model (SimCLR): The model to be fine tuned.
        device (str): The current torch device string.
        ts (list[Any]): A list of the transforms to be applied.
        num_epochs (int): The number of epochs to train over.
        learning_rate (float): The learning rate of the optimiser.
        split (float): The portion of the dataset to be sampled.
    """

    model.train()

    dataloader = FineTuningLoader(batch_size=32, num_workers=4).build(
        ts, split, "finetune"
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        total_loss = 0.0
        for i, (images, masks) in enumerate(dataloader):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            masks = label_segmentations(masks)

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

            if i % 40 == 0:
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

    print("Fine tuning complete.")
