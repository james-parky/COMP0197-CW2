import torch.nn as nn
from transformers import ViTMAEConfig


# Define the ViTMAE configuration
VIT_MAE_CONFIG = ViTMAEConfig(
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


class SegmentationHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.conv = nn.Conv2d(hidden_size, 1, kernel_size=1)
        self.upsample = nn.Upsample(
            scale_factor=8, mode="bilinear", align_corners=False
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)
        return x


class SegmentationModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.vit
        # Default hidden_size = 768
        hidden_size = self.encoder.config.hidden_size
        self.segmentation_head = SegmentationHead(hidden_size)

    def forward(self, pixel_values):
        outputs_encoder = self.encoder(pixel_values, return_dict=True)

        # last_hidden_state shape is (batch_size, sequence_length, hidden_size)
        # sequence_length is the number of patches
        last_hidden_state = outputs_encoder.last_hidden_state
        last_hidden_state = last_hidden_state[:, 1:]  # remove the [CLS] token

        # (batch_size,seq_length = 64, hidden_size = 768)\
        batch_size, sequence_length, hidden_size = last_hidden_state.shape
        height = width = int(sequence_length**0.5)

        last_hidden_state = last_hidden_state.permute(0, 2, 1)
        last_hidden_state2 = last_hidden_state.reshape(
            batch_size, hidden_size, height, width
        )
        segmentation_outputs = self.segmentation_head(last_hidden_state2)
        return segmentation_outputs
