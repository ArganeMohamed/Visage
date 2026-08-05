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
