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

