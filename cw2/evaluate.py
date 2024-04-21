from fine_tuning_loader import FineTuningLoader
import torch
from torch import nn
from label_segmentations import label_segmentations


def evaluate(model, device, ts, num_epochs, learning_rate, split):
    """
    Evaluate the given model on a sample of size split, of the Oxford-IIIT Pet dataset.
    Print periodic and final average loss, and the Intersection over Union accuracy.

    Args:
        model (SimCLR): The model to be fine tuned.
        device (str): The current torch device string.
        ts (list[Any]): A list of the transforms to be applied.
        num_epochs (int): The number of epochs to train over.
        learning_rate (float): The learning rate of the optimiser.
        split (float): The portion of the dataset to be sampled.
    """

    model.eval()

    dataloader = FineTuningLoader(batch_size=32, num_workers=4).build(
        ts, split, "eval"
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        total_loss = 0.0
        ious = []
        for i, (images, masks) in enumerate(dataloader):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            masks = label_segmentations(masks)
            inter = torch.logical_and(masks, outputs)
            uni = torch.logical_or(masks, outputs)

            ious.append(torch.count_nonzero(inter) / torch.count_nonzero(uni))

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.zero_grad()
            total_loss += loss.item()

            if i % 40 == 0:
                print(
                    (
                        f"[Epoch {epoch + 1}/{num_epochs},"
                        f"Batch {i + 1}/{len(dataloader)}]:"
                        f"Loss = {loss.item():.5f}"
                    )
                )

        avg_loss = total_loss / len(dataloader)
        print(
            f"[Epoch: {epoch + 1}/{num_epochs}]: Average Loss = {avg_loss:.5f}"
        )

        accuracy = torch.mean(torch.as_tensor(ious))
        print(f"[Epoch: {epoch + 1}/{num_epochs}]: IoU Accuracy = {accuracy}")

    print("Evaluation complete.")
