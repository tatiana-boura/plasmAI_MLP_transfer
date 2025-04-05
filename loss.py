import torch


class WeightedMSE:

    def __init__(self, reduction, outputs_points, device):
        self.reduction = reduction
        outputs_idx = [point-1 for point in outputs_points]
        self.weights = torch.tensor([2, 2, 2, 2, 1, 0.8, 0.5, 0.5, 0.1, 0.1]).to(device)
        self.weights = self.weights[outputs_idx]

    def __call__(self, input, target):

        squared_error = (input - target) ** 2 * self.weights

        if self.reduction == 'mean':
            return squared_error.mean()
        elif self.reduction == 'sum':
            return squared_error.sum()
        else:
            return squared_error

