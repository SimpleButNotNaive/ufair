import torch
from torch.nn import functional as F

def hinge_loss(Y, Y_hat, reduction='mean', weights=None):

    if weights == None:
        weights = torch.ones_like(Y)

    # Convert 0/1 label into -1/+1 label
    Y = 2 * Y - 1

    if reduction == 'mean':
        return (F.relu(1 - Y * Y_hat) * weights).sum() / weights.sum()
    elif reduction == 'sum':
        return (F.relu(1 - Y * Y_hat) * weights).sum()
    else:
        raise ValueError("Invalid reduction method!")

def calculate_dp_obj(Y_hat, Y, G):
    group_1_pr = hinge_loss(torch.zeros_like(Y_hat), Y_hat, weights= G)
    group_0_pr = hinge_loss(torch.zeros_like(Y_hat), Y_hat, weights= (1 - G))

    return torch.abs(group_1_pr - group_0_pr)

def calculate_eo_obj(Y_hat, Y, G):
    
    group_1_tpr = hinge_loss(1 - Y, Y_hat, weights= G * Y)
    group_0_tpr = hinge_loss(1 - Y, Y_hat, weights= (1 - G) * Y)

    return torch.abs(group_1_tpr - group_0_tpr)

def calculate_eodds_obj(Y_hat, Y, G):

    group_1_tpr = hinge_loss(1 - Y, Y_hat, weights= G * Y)
    group_0_tpr = hinge_loss(1 - Y, Y_hat, weights= (1 - G) * Y)

    group_1_fpr = hinge_loss(Y, Y_hat, weights= G * (1 - Y))
    group_0_fpr = hinge_loss(Y, Y_hat, weights= (1 - G) * (1 - Y))

    return torch.abs(group_1_tpr - group_0_tpr) + torch.abs(group_1_fpr - group_0_fpr)