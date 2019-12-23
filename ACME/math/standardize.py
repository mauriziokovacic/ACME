from .demean import *


def standardize(tensor, dim=-1):
    """
    Converts the tensor values into a Gaussian distribution

    Parameters
    ----------
    tensor : Tensor
        the input tensor
    dim : int or list or tuple (optional)
        the dimension(s) along the operation is performed (default is -1)

    Returns
    -------
    Tensor
        the output tensor
    """

    std = torch.std(tensor, dim=dim, keepdim=True)
    return demean(tensor, dim=dim) / std
