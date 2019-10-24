import torch


def mean_grad(model):
    g = []
    for p in model.parameters():
        if p.grad:
            g += [p.grad.abs().mean()]
        else:
            g += [0]
    return torch.tensor(g, dtype=torch.float, device='cpu').mean()
