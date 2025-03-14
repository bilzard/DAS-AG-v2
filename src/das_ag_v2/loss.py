import torch


def total_variation_loss(image):
    """
    Total Variation (TV) Loss
    :param image: (1, 3, H, W)
    :return: TV Loss
    """
    dx = torch.diff(image, dim=2).abs().mean()
    dy = torch.diff(image, dim=3).abs().mean()
    return dx + dy


def l1_loss(image):
    """
    L1 Regularization
    :param image: (1, 3, H, W)
    :return: L1 Loss
    """
    return image.abs().mean()
