import argparse
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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/data/matthew/cycle-diffusion/stargan-v2/data/test/cat/*.png",
    )
    parser.add_argument(
        "--source_ckpt", type=str, default="ckpts/cat_ema_0.9999_050000.pt"
    )
    parser.add_argument("--target_ckpt", type=str, default="ckpts/afhq_dog_4m.pt")
    parser.add_argument("--write_file", type=str, default="cyclediffusion.png")

    args = parser.parse_args()
    print(args)
    return args


def load_image(path, device):
    img = cv2.imread(path)
    img = cv2.resize(img, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.tensor(img).permute(2, 0, 1).to(device).float() / 255.0
    img = (img * 2.0 - 1.0).clip(-1, 1)
    return img


def load_images(dir, device, num_images=4):
    fnames = glob(dir)[:num_images]
    imgs = []
    for fname in fnames:
        imgs.append(load_image(fname, device))
    return torch.stack(imgs)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = get_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if os.path.isfile(args.write_file):
        os.remove(args.write_file)

    # Load source model
    source_model, diffusion = create_model_and_diffusion(
        **model_and_diffusion_defaults()
    )
    source_model.load_state_dict(torch.load(args.source_ckpt))
    source_model = source_model.to(device)
    source_model.eval()
    # Load target model
    target_model = deepcopy(source_model)
    target_model.load_state_dict(torch.load(args.target_ckpt))
    target_model = target_model.to(device)
    target_model.eval()

    # Cycle consistency test
    print("encoding...")
    imgs = load_images(args.data_dir, device, num_images=args.num_samples)
    source_z = diffusion.encode(source_model, imgs, args.timesteps, device)
    print("decoding...")
    target_out = diffusion.decode(target_model, source_z, args.timesteps, device)
    print("encoding...")
    target_z = diffusion.encode(target_model, target_out, args.timesteps, device)
    print('decoding...')
    source_out = diffusion.decode(source_model, target_z, args.timesteps, device)

    save_image(
        make_grid(
            torch.concatenate([imgs, target_out, source_out], dim=0),
            value_range=(-1, 1),
            normalize=True,
            nrow=args.num_samples,
        ),
        args.write_file,
    )
