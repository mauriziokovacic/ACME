import torch


def mean_grad(model):
    """
    Return the mean gradient of the given model

    Parameters
    ----------
    model : torch.nn.Module
        the model to evaluate

    Returns
    -------
    Tensor
        the (1,) mean gradient tensor
    """

    g = []
    for p in model.parameters():
        if p.grad is None:
            g += [0]
        else:
            g += [p.grad.abs().mean()]
    return torch.tensor(g, dtype=torch.float, device='cpu').mean()
