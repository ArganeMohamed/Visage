import torch
from tqdm import tqdm


class Sampler:
    def __init__(self, model, diffusion, config):
        self.model     = model
        self.diffusion = diffusion
        self.config    = config
        self.device    = config['device']

    @torch.no_grad()
    def sample(self, n_samples=16):
        self.model.eval()

        x = torch.randn(
            n_samples,
            self.config['channels'],
            self.config['image_size'],
            self.config['image_size']
        ).to(self.device)

        for t in tqdm(reversed(range(self.diffusion.timesteps)), desc="Sampling", total=self.diffusion.timesteps):
            t_batch = torch.full((n_samples,), t, device=self.device, dtype=torch.long)

            predicted_noise = self.model(x, t_batch)

            alpha              = self.diffusion.alphas[t].to(self.device)
            alpha_cumprod      = self.diffusion.alphas_cumprod[t].to(self.device)
            alpha_cumprod_prev = self.diffusion.alphas_cumprod[t-1].to(self.device) if t > 0 else torch.tensor(1.0).to(self.device)
            beta               = self.diffusion.betas[t].to(self.device)

            x0_pred = (x - torch.sqrt(1 - alpha_cumprod) * predicted_noise) / torch.sqrt(alpha_cumprod)
            x0_pred = x0_pred.clamp(-1, 1)

            if t > 0:
                noise = torch.randn_like(x)
                posterior_variance = beta * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod)
                x = (
                    torch.sqrt(alpha_cumprod_prev) * beta / (1 - alpha_cumprod) * x0_pred +
                    torch.sqrt(alpha) * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod) * x +
                    torch.sqrt(posterior_variance) * noise
                )
            else:
                x = x0_pred

        self.model.train()
        return x
