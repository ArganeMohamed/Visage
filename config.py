def get_config(**overrides):
    config = {
        'image_size'      : 64,
        'channels'        : 3,

        'timesteps'       : 1000,
        'noise_schedule'  : 'cosine',

        'patch_size'      : 4,
        'dim'             : 384,
        'depth'           : 8,
        'num_heads'       : 6,
        'mlp_ratio'       : 4.0,

        'batch_size'      : 64,
        'learning_rate'   : 0.0001,
        'epochs'          : 100,
        'grad_clip'       : 1.0,
        'ema_decay'       : 0.9999,

        'device'          : 'cuda',
        'mixed_precision' : True,
        'num_workers'     : 4,
    }

    config.update(overrides)

    assert config['image_size'] % config['patch_size'] == 0, "image_size must be divisible by patch_size"
    assert config['dim'] % config['num_heads'] == 0, "dim must be divisible by num_heads"
    assert config['dim'] % 4 == 0, "dim must be divisible by 4 (needed for 2D sin-cos position embedding)"

    return config