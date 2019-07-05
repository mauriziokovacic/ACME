import torch


def normw(W):
    """
    Normalizes the given weights

    Parameters
    ----------
    W : Tensor
        the (N,H,) weights matrix

    Returns
    -------
    Tensor
        the (N,H,) normalized weights matrix
    """

    return W/torch.sum(W, 1, keepdim=True)
