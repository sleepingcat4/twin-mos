import torch
import torch.nn as nn
import torch.nn.functional as F


class MetricRegressionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z1, z2, score1, score2):
        distance = F.pairwise_distance(z1, z2)
        target = torch.abs(score1 - score2)

        return torch.mean((distance - target) ** 2)