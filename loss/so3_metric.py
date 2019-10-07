import torch


def so3_metric(M):
    return torch.mean(torch.norm(torch.matmul(M, torch.transpose(M, -1, -2))-torch.eye(3), dim=(-2,-1), keepdim=True))
