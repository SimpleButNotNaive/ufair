import torch
def calculate_acc(Y_hat, Y):

    Y_prediction = torch.where(Y_hat > 0, torch.ones_like(Y_hat), torch.zeros_like(Y_hat))
    acc = torch.where(Y_prediction == Y, torch.ones_like(Y_hat), torch.zeros_like(Y_hat)).mean().item()
    return acc

def calculate_dp(Y_hat, Y, G):
    Y_prediction = torch.where(Y_hat > 0, torch.ones_like(Y_hat), torch.zeros_like(Y_hat))
    group_1_pr = Y_prediction[G == 1].mean()
    group_0_pr = Y_prediction[G == 0].mean()

    # return min(group_0_pr / (group_1_pr + 1e-6), group_1_pr / (group_0_pr + 1e-6)).item()
    return torch.abs(group_1_pr - group_0_pr).item()


def calculate_eo(Y_hat, Y, G):

    Y_prediction = torch.where(Y_hat > 0, torch.ones_like(Y_hat), torch.zeros_like(Y_hat))
    group_1_tpr = Y_prediction[(Y == 1) & (G == 1)].mean()
    group_0_tpr = Y_prediction[(Y == 1) & (G == 0)].mean()

    return torch.abs(group_1_tpr - group_0_tpr).item()

def calculate_eodds(Y_hat, Y, G):

    Y_prediction = torch.where(Y_hat > 0, torch.ones_like(Y_hat), torch.zeros_like(Y_hat))
    group_1_tpr = Y_prediction[(Y == 1) & (G == 1)].mean()
    group_0_tpr = Y_prediction[(Y == 1) & (G == 0)].mean()

    group_1_fpr = Y_prediction[(Y == 0) & (G == 1)].mean()
    group_0_fpr = Y_prediction[(Y == 0) & (G == 0)].mean()

    return torch.abs(group_1_tpr - group_0_tpr).item() + torch.abs(group_1_fpr - group_0_fpr).item()