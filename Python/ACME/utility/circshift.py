import torch

def circshift(tensor,k,dim=None):
    """
    Circularly shifts the input tensor k times along the given dimension

    Parameters
    ----------
    tensor : Tensor
        the input tensor
    k : int
        the number of shifts to perform
    dim : int (optional)
        the dimension along the shift is performed (default is None)
    """

    return torch.roll(tensor,k,dims=dim)
