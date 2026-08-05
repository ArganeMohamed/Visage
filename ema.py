from copy import deepcopy
import torch


class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.ema_model = deepcopy(model)
        self.ema_model.eval()

        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            for ema_param, model_param in zip(
                self.ema_model.parameters(),
                model.parameters()
            ):
                ema_param.data = (
                    self.decay * ema_param.data +
                    (1 - self.decay) * model_param.data
                )

    def get_model(self):
        return self.ema_model


if __name__ == '__main__':
    import torch.nn as nn

    # tiny dummy model just to verify EMA math, not the real DiT
    model = nn.Linear(4, 4)
    ema = EMA(model, decay=0.9)

    # snapshot EMA weight before any update
    before = ema.ema_model.weight.data.clone()

    # change the "trained" model's weights drastically
    with torch.no_grad():
        model.weight.fill_(5.0)

    ema.update(model)
    after = ema.ema_model.weight.data.clone()

    print('EMA weight before update (should be original init, not 5.0):')
    print(before[0])
    print('EMA weight after one update (should move toward 5.0 but not reach it):')
    print(after[0])
    print('Moved toward target:', bool((after - before).abs().sum() > 0))
    print('Did NOT jump all the way to 5.0:', bool((after == 5.0).sum() == 0))

    # EMA model should not require grad
    requires_grad = any(p.requires_grad for p in ema.ema_model.parameters())
    print('EMA model requires_grad (should be False):', requires_grad)