import torch


def weight_valence(W):
    """
    Returns the number of non zero weights per vertex

    Parameters
    ----------
    W : Tensor
        the (N,H,) weights matrix

    Returns
    -------
    LongTensor
        the (N,1,) weights valence tensor
    """

    return torch.sum(W != 0, 1, keepdim=True).to(dtype=torch.long)
