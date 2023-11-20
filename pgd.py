import torch

def normalize_grad(grads, loss):
    grads_norm = 0
    for g in grads:
        grads_norm += g.pow(2).sum()
    return [g / (loss * torch.sqrt(grads_norm)) for g in grads]
import numpy as np

def projection2simplex(y):
    """
    Given y, it solves argmin_z |y-z|_2 st \sum z = 1 , 1 >= z_i >= 0 for all i
    """
    y = y.numpy()
    m = len(y)
    sorted_y = np.flip(np.sort(y), axis=0)
    tmpsum = 0.0
    tmax_f = (np.sum(y) - 1.0)/m
    for i in range(m-1):
        tmpsum+= sorted_y[i]
        tmax = (tmpsum - 1)/ (i+1.0)
        if tmax > sorted_y[i+1]:
            tmax_f = tmax
            break
    return torch.tensor(np.maximum(y - tmax_f, np.zeros(y.shape)), dtype=torch.float)

def solve_min_norm(obj_grads, alpha_lr, update_step=10):
    alpha = torch.ones(len(obj_grads), dtype=torch.float) / len(obj_grads)

    # print("=======================")
    for _ in range(update_step):
        weighted_sum = torch.matmul(torch.t(obj_grads), alpha)
        alpha_grad = 2 * torch.matmul(obj_grads, weighted_sum)

        alpha -= alpha_lr * alpha_grad
        # print("alpha_grad: ", alpha_grad, "alpha_raw", alpha)
        alpha = projection2simplex(alpha)
        # print(torch.norm(weighted_sum))

    min_norm = torch.norm(torch.matmul(torch.t(obj_grads), alpha))
    return alpha, min_norm