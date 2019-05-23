import torch

def istorch(A):
    """
    Returns whether or not the input is a PyTorch tensor

    Parameters
    ----------
    obj : object
        any object

    Returns
    -------
    bool
        True if the input is a PyTorch Tensor, False otherwise
    """

    return isinstance(obj,torch.Tensor)
