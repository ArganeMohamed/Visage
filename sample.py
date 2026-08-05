import argparse
import torch

from config import get_config
from models.dit import DiT
from diffusion import Diffusion
from ema import EMA
from sampler import Sampler
from utils import save_samples


def generate(checkpoint_path, output_path='./outputs', n_samples=16, **config_overrides):
    config = get_config(**config_overrides)
    device = config['device']

    model = DiT(config).to(device)
    ema   = EMA(model, decay=config['ema_decay'])

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    ema.ema_model.load_state_dict(checkpoint['ema'])
    epoch = checkpoint['epoch']
    print(f"Loaded checkpoint from epoch {epoch}")

    diffusion = Diffusion(config)
    sampler   = Sampler(ema.get_model(), diffusion, config)  # sample from EMA weights, not raw model

    print(f"Generating {n_samples} faces...")
    samples = sampler.sample(n_samples=n_samples)

    img_path = save_samples(samples, epoch, output_path)
    print(f"Saved -> {img_path}")
    return img_path

