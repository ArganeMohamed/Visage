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


if __name__ == '__main__':
    from config import get_config

    config = get_config(device='cpu')  # schedule math itself doesn't need GPU
    diffusion = Diffusion(config)

    print('Betas shape:', diffusion.betas.shape)
    print('Betas[0]:', diffusion.betas[0].item(), '| Betas[-1]:', diffusion.betas[-1].item())
    print('Betas monotonically increasing:', bool((diffusion.betas[1:] >= diffusion.betas[:-1]).all()))

    print('Alphas_cumprod[0]:', diffusion.alphas_cumprod[0].item(), '(expect close to 1.0)')
    print('Alphas_cumprod[-1]:', diffusion.alphas_cumprod[-1].item(), '(expect close to 0.0)')

    # add_noise sanity check: at t=0 should be almost clean, at t=999 should be almost pure noise
    x0 = torch.randn(4, 3, 64, 64)  # pretend clean image
    t_low  = torch.tensor([0, 0, 0, 0])
    t_high = torch.tensor([999, 999, 999, 999])

    noisy_low, _  = diffusion.add_noise(x0, t_low)
    noisy_high, _ = diffusion.add_noise(x0, t_high)

    diff_low  = (noisy_low - x0).abs().mean().item()
    diff_high = (noisy_high - x0).abs().mean().item()
    print(f'Mean abs diff from original at t=0:   {diff_low:.4f}  (expect small)')
    print(f'Mean abs diff from original at t=999: {diff_high:.4f} (expect large)')
    print('t=999 noisier than t=0:', diff_high > diff_low)