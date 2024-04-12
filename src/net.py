"""
Module implementing an image segmentation class based on Yipeng's tutorial written in 
tensorflow style Python, available at 
https://github.com/james-parky/COMP0197/blob/main/tutorials/img_sgm/network_tf.py
"""
from typing import Optional, Tuple
from torch import nn
import torch


class ResUNet(nn.Module):
    """
    Image segmentation class based on the ResUNet architecture, completing up and
    downsampling during a forward pass.
    """

    def __init__(self, init_ch: int = 32, num_levels: int = 3, out_ch: int = 1):
        super().__init__()
        self.first_layer = self._conv2d_layer(init_ch, in_ch=3)
        self.encoder = nn.ModuleList(
            [
                self._resnet_block(2**i * init_ch, t="down")
                for i in range(num_levels)
            ]
        )
        self.encoder.append(
            self._resnet_block(2**num_levels * init_ch, t="none")
        )
        self.decoder = nn.ModuleList(
            [
                self._resnet_block(2**i * init_ch, t="up")
                for i in range(num_levels, 0, -1)
            ]
        )
        self.decoder.append(self._resnet_block(init_ch, t="none"))

        self.out_layer = self._conv2d_layer(out_ch, 16, is_output=True)
        self.input_shape = (64, 64, 3)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Implement a forward pass in the ResUNet architecture with up and downsampling.

        Args:
            inputs (torch.Tensor): The image being passed through the network.

        Returns:
            (torch.Tensor): The output of the net.
        """
        x = self.first_layer(inputs)
        skips = []
        for down in self.encoder[:-1]:
            x = down(x)
            skips.append(x)
        x = self.encoder[-1](x)
        for up, skip in zip(self.decoder[:-1], reversed(skips)):
            x = self._resize_to(x, skip) + skip
            x = up(x)
        x = self._resize_to(x, inputs)
        x = self.decoder[-1](x)
        x = self.out_layer(x)
        return x

    def _conv2d_layer(
        self, ch: int, in_ch: Optional[int] = None, is_output: bool = False
    ) -> None:
        if in_ch is None:
            in_ch = ch
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=ch,
                kernel_size=3,
                padding=1,
                bias=is_output,
            ),
            nn.Identity() if is_output else nn.BatchNorm2d(ch),
            nn.Sigmoid() if is_output else nn.ReLU(),
        )

    def _resnet_block(
        self, ch: int, t: str
    ) -> torch.nn.modules.container.Sequential:
        layers = [
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        ] * 2
        if t == "down":
            layers.extend(
                [
                    nn.Conv2d(ch, ch * 2, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(ch * 2),
                    nn.ReLU(inplace=True),
                ]
            )
            ch *= 2
        elif t == "up":
            layers.extend(
                [
                    nn.ConvTranspose2d(
                        ch,
                        ch // 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(ch // 2),
                    nn.ReLU(inplace=True),
                ]
            )
            ch //= 2
        return nn.Sequential(*layers)

    def _resize_to(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.shape[2:4] == y.shape[2:4]:
            return x
        if any(abs(x.shape[i] - y.shape[i]) > 1 for i in [2, 3]):
            raise Warning("padding/cropping more than 1")
        return nn.functional.interpolate(
            x,
            size=(y.shape[2], y.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
