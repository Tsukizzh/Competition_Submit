import torch
import torch.nn as nn

class RootMeanSquaredLogarithmicError(nn.Module):
    def __init__(self):
        super(RootMeanSquaredLogarithmicError, self).__init__()

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.mean((torch.log1p(y_true) - torch.log1p(y_pred))**2))

class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight=None):
        super(WeightedBCEWithLogitsLoss, self).__init__()
        if pos_weight is None:
            pos_weight = 1.5
        self.pos_weight = torch.tensor([float(pos_weight)])

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        loss_fct = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(y_pred.device))
        return loss_fct(y_pred, y_true)