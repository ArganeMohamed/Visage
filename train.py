import os
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from config import get_config
from dataset import get_dataloader
from models.dit import DiT
from diffusion import Diffusion
from ema import EMA
from sampler import Sampler
from scheduler import WarmupCosineScheduler
from utils import save_samples, save_checkpoint, load_checkpoint


def train(
    data_path,
    output_path='./outputs',
    checkpoint_path=None,
    resume=True,
    sample_every=10,
    log_every=100,
    max_batches_per_epoch=None,   # testing-only: caps steps/epoch, leave None for real training
    **config_overrides
):
    if checkpoint_path is None:
        checkpoint_path = f"{output_path}/checkpoints"

    config = get_config(**config_overrides)
    device = config['device']

    print(f"Device: {device}")
    print(f"Config: {config}")

    dataloader = get_dataloader(data_path, config)
    print(f"Dataset: {len(dataloader.dataset)} images, {len(dataloader)} batches/epoch")

    model     = DiT(config).to(device)
    diffusion = Diffusion(config)
    ema       = EMA(model, decay=config['ema_decay'])
    sampler   = Sampler(ema.get_model(), diffusion, config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.01)
    loss_fn   = nn.MSELoss()
    scaler    = GradScaler('cuda', enabled=(config['mixed_precision'] and device == 'cuda'))

    steps_per_epoch = max_batches_per_epoch or len(dataloader)
    total_steps = steps_per_epoch * config['epochs']
    warmup_steps = min(1000, max(1, total_steps // 10))
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params/1e6:.2f}M")

    start_epoch = 1
    if resume:
        os.makedirs(checkpoint_path, exist_ok=True)
        checkpoints = sorted([f for f in os.listdir(checkpoint_path) if f.endswith('.pt')])
        if checkpoints:
            latest = os.path.join(checkpoint_path, checkpoints[-1])
            print(f"Resuming from: {latest}")
            start_epoch = load_checkpoint(latest, model, ema, optimizer, device) + 1

    os.makedirs(f"{output_path}/logs", exist_ok=True)
    log_path = f"{output_path}/logs/loss.txt"

    for epoch in range(start_epoch, config['epochs'] + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, images in enumerate(dataloader):
            if max_batches_per_epoch and batch_idx >= max_batches_per_epoch:
                break

            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0])
            noisy_images, actual_noise = diffusion.add_noise(images, t)

            with autocast('cuda', enabled=(config['mixed_precision'] and device == 'cuda')):
                predicted_noise = model(noisy_images, t)
                loss = loss_fn(predicted_noise, actual_noise)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)

            total_loss += loss.item()
            num_batches += 1

            if batch_idx % log_every == 0:
                print(f"  Epoch {epoch:03d} | Batch {batch_idx:04d}/{len(dataloader)} | Loss: {loss.item():.4f} | LR: {scheduler.get_lr():.6f}")

        avg_loss = total_loss / max(1, num_batches)
        print(f"Epoch {epoch:03d} done -- Avg Loss: {avg_loss:.4f}")

        with open(log_path, 'a') as f:
            f.write(f"Epoch {epoch}, Loss: {avg_loss:.4f}\n")

        if epoch % sample_every == 0 or epoch == config['epochs']:
            samples = sampler.sample(n_samples=16)
            img_path  = save_samples(samples, epoch, output_path)
            ckpt_path = save_checkpoint(model, ema, optimizer, epoch, avg_loss, checkpoint_path)
            print(f"Saved samples -> {img_path}")
            print(f"Saved checkpoint -> {ckpt_path}")

    return model, ema