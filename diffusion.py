import numpy as np
import torch


class Diffusion:
    def __init__(self, config):
        self.timesteps = config['timesteps']
        self.device = config['device']

        if config['noise_schedule'] == 'cosine':
            self.betas = self._cosine_schedule()
        else:
            self.betas = self._linear_schedule()

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def _linear_schedule(self):
        return torch.linspace(1e-4, 0.02, self.timesteps)

    def _cosine_schedule(self):
        steps = self.timesteps + 1
        x = torch.linspace(0, self.timesteps, steps)
        alphas_cumprod = torch.cos(((x / self.timesteps) + 0.008) / 1.008 * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)

    def add_noise(self, x0, t):
        noise = torch.randn_like(x0)
        t_cpu = t.cpu()

        sqrt_alpha     = self.sqrt_alphas_cumprod[t_cpu].to(self.device)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t_cpu].to(self.device)

        sqrt_alpha     = sqrt_alpha[:, None, None, None]
        sqrt_one_minus = sqrt_one_minus[:, None, None, None]

        return sqrt_alpha * x0 + sqrt_one_minus * noise, noise

    def sample_timesteps(self, batch_size):
        return torch.randint(0, self.timesteps, (batch_size,), device=self.device)