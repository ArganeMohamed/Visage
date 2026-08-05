import math


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps):
        self.optimizer     = optimizer
        self.warmup_steps  = warmup_steps
        self.total_steps   = total_steps
        self.current_step  = 0
        self.base_lr       = optimizer.param_groups[0]['lr']

    def step(self):
        self.current_step += 1

        if self.current_step < self.warmup_steps:
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            progress = (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))

        lr = max(lr, 1e-7)
        for group in self.optimizer.param_groups:
            group['lr'] = lr

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


if __name__ == '__main__':
    import torch

    dummy_param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([dummy_param], lr=1e-4)

    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=10, total_steps=100)

    print('Initial LR:', scheduler.get_lr(), '(expect 1e-4, unchanged before any step)')

    # step through warmup
    for _ in range(10):
        scheduler.step()
    lr_after_warmup = scheduler.get_lr()
    print('LR after warmup (step 10):', lr_after_warmup, '(expect close to 1e-4, warmup complete)')

    # step through the rest (cosine decay to the end)
    for _ in range(90):
        scheduler.step()
    lr_at_end = scheduler.get_lr()
    print('LR at final step (step 100):', lr_at_end, '(expect close to 0, cosine decayed to minimum)')

    print('LR increased during warmup then decreased after:', lr_after_warmup > 1e-5 and lr_at_end < lr_after_warmup)