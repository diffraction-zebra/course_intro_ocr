from math import pi

import torch
from torch import nn


class ArcFaceLoss():
    def __init__(self, num_classes, margin=0.5, scale=64):
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.threshold = torch.cos(torch.tensor(pi - margin))
        self.cos_m = torch.cos(torch.tensor(margin))
        self.sin_m = torch.sin(torch.tensor(margin))

        self.safe_margin = self.sin_m * margin

    def __call__(self, y_pred, y_true):

        assert len(y_pred.shape) == 2, 'only [batch : labels] implementation'
        assert len(y_pred) == len(y_true)

        # Calculate the cosine value of theta + margin.
        cos_t = y_pred
        sin_t = torch.sqrt(1 - torch.square(cos_t))

        cos_t_margin = torch.where(cos_t > self.threshold,
                                cos_t * self.cos_m - sin_t * self.sin_m,
                                cos_t - self.safe_margin)

        # replace true logits with margin logits
        mask = (torch.arange(self.num_classes, dtype=int).to(y_true.device) == torch.unsqueeze(y_true, 1))
        logits = torch.where(mask,
                             cos_t_margin,
                             cos_t)
        logits = logits * self.scale


        losses = nn.CrossEntropyLoss()(logits, y_true)

        return losses
