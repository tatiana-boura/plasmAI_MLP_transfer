import torch


class WeightedMSE:

    def __init__(self, reduction, device):
        self.reduction = reduction
        self.weights = torch.tensor([2, 2, 2, 2, 1, 0.8, 0.5, 0.5, 0.1, 0.1]).to(device)

    def __call__(self, input, target):

        squared_error = (input - target) ** 2 * self.weights

        if self.reduction == 'mean':
            return squared_error.mean()
        elif self.reduction == 'sum':
            return squared_error.sum()
        else:
            return squared_error


class HuberLoss:
    def __init__(self, delta: float = 1.0):
        self.delta = delta

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Compute the absolute error
        error = torch.abs(y_true - y_pred)

        # Calculate Huber loss based on the threshold delta
        loss = torch.where(error <= self.delta,
                           0.5 * error ** 2,  # Squared loss
                           self.delta * (error - 0.5 * self.delta))  # Absolute loss

        return loss.mean()
