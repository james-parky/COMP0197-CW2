""" Pretraining ViTMAE model """

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from transformers import (
    ViTMAEForPreTraining,
    ViTMAEConfig,
)  # pylint: disable=import-error
from unsupervised_loader import UnsupervisedLoader

# Define the ViTMAE configuration
config = ViTMAEConfig(
    image_size=64,
    patch_size=4,
    num_channels=3,
    encoder_layers=6,
    decoder_layers=4,
    encoder_num_heads=8,
    decoder_num_heads=8,
    encoder_hidden_size=64,
    decoder_hidden_size=64,
    mlp_ratio=4,
    norm_pix_loss=True,
)

# Create the ViTMAEForPreTraining model
model = ViTMAEForPreTraining(config)

# Define the data transforms
pretraining_transforms = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

# # Download and create the pretraining dataset
# dataset = CIFAR10(root="./data", train=True, download=True, transform=pretraining_transforms)
# pretraining_dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
#
pretraining_dataloader = UnsupervisedLoader(
    img_size=(64, 64), batch_size=32
).build(data_augmentation=[pretraining_transforms])
# Set up the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Pretraining loop
NUM_EPOCHS = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

total_steps = len(pretraining_dataloader) * NUM_EPOCHS
CURRENT_STEP = 0

for epoch in range(NUM_EPOCHS):
    model.train()
    for batch_idx, batch in enumerate(pretraining_dataloader):
        images, _ = batch  # Ignore the labels during pretraining
        images = images.to(device)
        outputs = model(pixel_values=images)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        CURRENT_STEP += 1
        if (batch_idx + 1) % 1 == 0:
            print(
                f"Epoch [{epoch+1}/{NUM_EPOCHS}] \
                  Batch [{batch_idx+1}/{len(pretraining_dataloader)}] Loss: {loss.item():.4f}"
            )
        if CURRENT_STEP % 1000 == 0:
            progress = (CURRENT_STEP / total_steps) * 100
            print(f"Progress: {progress:.2f}%")

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Loss: {loss.item():.4f}")

# Save the pretrained model
PRETRAINED_MODEL_PATH = "./pretrained_model.pth"
torch.save(model.state_dict(), PRETRAINED_MODEL_PATH)
print("Pretraining completed. Model saved.")
