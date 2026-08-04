import torch
import torch.nn as nn

from models.layers import (
    get_2d_sincos_pos_embed,
    TimestepEmbedder,
    PatchEmbed,
    Attention,
)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn  = Attention(dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(approximate='tanh'),
            nn.Linear(hidden, dim)
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)

        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, dim, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class DiT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_size   = config['patch_size']
        self.out_channels = config['channels']
        self.dim          = config['dim']

        self.patch_embed = PatchEmbed(config['image_size'], self.patch_size, config['channels'], self.dim)
        self.grid_size    = self.patch_embed.grid_size

        self.t_embedder = TimestepEmbedder(self.dim)

        pos_embed = get_2d_sincos_pos_embed(self.dim, self.grid_size)
        self.register_buffer(
            'pos_embed',
            torch.from_numpy(pos_embed).float().unsqueeze(0),
            persistent=False
        )

        self.blocks = nn.ModuleList([
            DiTBlock(self.dim, config['num_heads'], config['mlp_ratio'])
            for _ in range(config['depth'])
        ])
        self.final_layer = FinalLayer(self.dim, self.patch_size, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def basic_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(basic_init)

        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.patch_embed.proj.bias, 0)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)


        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        B = x.shape[0]
        p, g, C = self.patch_size, self.grid_size, self.out_channels

        x = x.reshape(B, g, g, p, p, C)
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(B, C, g * p, g * p)

    def forward(self, x, t):
        x = self.patch_embed(x) + self.pos_embed
        c = self.t_embedder(t)

        for block in self.blocks:
            x = block(x, c)

        x = self.final_layer(x, c)
        return self.unpatchify(x)


if __name__ == '__main__':
    from config import get_config

    config = get_config()
    model = DiT(config)

    x = torch.randn(2, config['channels'], config['image_size'], config['image_size'])
    t = torch.randint(0, config['timesteps'], (2,))

    out = model(x, t)
    print('Input shape:', x.shape)
    print('Output shape:', out.shape)
    assert out.shape == x.shape, f"MISMATCH: {out.shape} vs {x.shape}"
    print('Shapes match!')

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params / 1e6:.2f}M')