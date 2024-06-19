import os
import random
from copy import deepcopy
from glob import glob

import cv2
import numpy as np
import torch
from guided_diffusion.script_util import (create_model_and_diffusion,
                                          model_and_diffusion_defaults)
from torchvision.utils import make_grid, save_image

SEED = 42
N = 4


def load_image(path, device):
    img = cv2.imread(path)
    img = cv2.resize(img, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.tensor(img).permute(2, 0, 1).to(device).float() / 255.0
    img = (img * 2.0 - 1.0).clip(-1, 1)
    return img


def load_images(device, num_images=4):
    fnames = glob("/data/matthew/cycle-diffusion/stargan-v2/data/test/cat/*.png")[
        :num_images
    ]
    imgs = []
    for fname in fnames:
        imgs.append(load_image(fname, device))
    return torch.stack(imgs)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)
    os.remove("cyclediffusion.png")

    # Load source model
    source_model, diffusion = create_model_and_diffusion(
        **model_and_diffusion_defaults()
    )
    source_model.load_state_dict(torch.load("ckpts/cat_ema_0.9999_050000.pt"))
    source_model = source_model.to(device)
    source_model.eval()
    # Load target model
    target_model = deepcopy(source_model)
    target_model.load_state_dict(torch.load("ckpts/cat_ema_0.9999_050000.pt"))
    target_model = target_model.to(device)
    target_model.eval()

    T = 1000
    print("encoding...")
    imgs = load_images(device, num_images=N)
    z = diffusion.encode(source_model, imgs, T, device)
    print("generating...")
    out = diffusion.generate(target_model, z, T, device)

    save_image(
        make_grid(
            torch.concatenate([imgs, out], dim=0),
            value_range=(-1, 1),
            normalize=True,
            nrow=N,
        ),
        "cyclediffusion.png",
    )
