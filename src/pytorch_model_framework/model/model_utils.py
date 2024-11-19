import torch


def focal_loss(raw_loss: torch.Tensor, alpha: float, gamma: float) -> torch.Tensor:
    focal_loss = alpha * (1 - torch.exp(-raw_loss)) ** gamma * raw_loss  # per batch focal loss
    return focal_loss


def dice_loss(y_hat: torch.Tensor, y: torch.Tensor):
    intersection = torch.sum(y_hat * y)
    union = torch.sum(y_hat) + torch.sum(y)
    dice_loss = 1 - (2 * intersection / union)
    return dice_loss
