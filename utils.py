import os
import torch
import matplotlib.pyplot as plt


def save_samples(images, epoch, output_path):
    output_dir = f"{output_path}/samples"
    os.makedirs(output_dir, exist_ok=True)

    images = (images.clamp(-1, 1) + 1) / 2
    images = images.cpu().permute(0, 2, 3, 1).numpy()

    n    = len(images)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten() if n > 1 else [axes]

    for i, img in enumerate(images):
        axes[i].imshow(img)
        axes[i].axis('off')

    for i in range(n, len(axes)):
        axes[i].axis('off')

    plt.suptitle(f'Epoch {epoch}', fontsize=14)
    plt.tight_layout()

    save_path = f"{output_dir}/epoch_{epoch:04d}.png"
    plt.savefig(save_path)
    plt.close()
    return save_path


def save_checkpoint(model, ema, optimizer, epoch, loss, checkpoint_path):
    os.makedirs(checkpoint_path, exist_ok=True)
    path = f"{checkpoint_path}/checkpoint_epoch_{epoch:04d}.pt"
    torch.save({
        'epoch'     : epoch,
        'model'     : model.state_dict(),
        'ema'       : ema.ema_model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'loss'      : loss,
    }, path)
    return path


def load_checkpoint(path, model, ema, optimizer, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    ema.ema_model.load_state_dict(checkpoint['ema'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch']