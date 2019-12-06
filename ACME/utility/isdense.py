import torch


def isdense(*obj):
    """
    Returns whether or not the inputs are PyTorch dense tensors

    Parameters
    ----------
    *obj : object...
        a sequence of objects

    Returns
    -------
    bool
        True if the inputs are PyTorch dense Tensors, False otherwise
    """

    return all([isinstance(o, torch.FloatTensor) for o in obj])
