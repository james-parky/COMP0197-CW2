import itertools
from sim_clr_model import SimCLR
from torchvision.models import resnet18
from torch import nn
from transformers import ViTMAEForPreTraining
from vit_mae_model import VIT_MAE_CONFIG, SegmentationModel
from sim_clr_train import sim_clr_train
from vit_mae_train import vit_mae_train
from unet_model import UNet
from unet_train import unet_train
from fine_tune import fine_tune
from torchvision.transforms import (
    Compose,
    Resize,
    ToTensor,
    Normalize,
    RandomGrayscale,
    RandomHorizontalFlip,
)
from evaluate import evaluate
import torch
from lightly.transforms.simclr_transform import SimCLRTransform


def powerset(s: list) -> list:
    """
    Return the powerset of s. Code found at https://stackoverflow.com/questions/1482308/how-to-get-all-subsets-of-a-set-powerset.
    itertools is part of the standard python library, but for some reason is not in the
    conda env. So we wrote this.

    Args:
        s (list): The set to return the powerset of.

    Returns:
        (list): The powerset of s.
    """
    p = []
    l = len(s)
    for i in range(1 << l):
        p.append([s[j] for j in range(l) if (i & (1 << j))])
    return p


if __name__ == "__main__":
    transforms = [RandomHorizontalFlip(p=0.5), RandomGrayscale(p=0.5)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    finetune_split = 0.5
    pretrain_transforms = [
        Resize((64, 64)),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]

    finetune_transforms = [
        Resize((64, 64)),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    for ts in powerset(transforms):
        print(f"Transforms Being Used: {ts}")

        sim_clr_model = SimCLR(
            nn.Sequential(*list(resnet18().children())[:-1])
        ).to(device)
        sim_clr_pretain_args = [
            device,
            Compose(ts + [SimCLRTransform(input_size=32, gaussian_blur=0.0)]),
            10,
            0.06,
        ]
        sim_clr_finetune_args = [
            device,
            Compose(ts + finetune_transforms),
            10,
            0.06,
            finetune_split,
        ]

        sim_clr_eval_args = sim_clr_finetune_args
        sim_clr_train(sim_clr_model, *sim_clr_pretain_args)
        fine_tune(sim_clr_model, *sim_clr_finetune_args)
        evaluate(sim_clr_model, *sim_clr_eval_args)

        vit_mae_model = ViTMAEForPreTraining(VIT_MAE_CONFIG)
        vit_mae_model.to(device)
        vit_mae_pretrain_args = [
            device,
            Compose(ts + pretrain_transforms),
            10,
            1e-4,
        ]
        vit_mae_finetune_args = [
            device,
            Compose(ts + finetune_transforms),
            10,
            1e-4,
            finetune_split,
        ]
        vit_mae_eval_args = vit_mae_pretrain_args
        vit_mae_train(vit_mae_model, *vit_mae_pretrain_args)
        vit_mae_model = SegmentationModel(vit_mae_model).to(device)
        fine_tune(vit_mae_model, *vit_mae_finetune_args)
        evaluate(vit_mae_model, *vit_mae_eval_args)

        unet_model = UNet().to(device)
        unet_train_args = [device, Compose(ts + finetune_transforms), 10, 1e-4]
        unet_eval_args = unet_train_args
        unet_train(unet_model, *unet_train_args)
        evaluate(unet_model, *unet_eval_args)
