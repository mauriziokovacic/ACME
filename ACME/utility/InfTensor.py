from .ConstantTensor import *
import math


def InfTensor(*size, dtype=torch.float, device='cuda:0'):
    """
    Returns a tensor containing only Inf values

    Parameters
    ----------
    *size : int...
        the shape of the tensor
    dtype : type (optional)
        the type of the Tensor (default is torch.float)
    device : str or torch.device (optional)
        the device to store the tensor to (default is 'cuda:0')

    Returns
    -------
    Tensor
        a tensor made out of occurrences of Inf
    """

    return ConstantTensor(math.inf, *size, dtype=dtype, device=device)


def inf_like(tensor):
    """
    Returns a tensor filled with Inf, shaped as the given tensor

    Parameters
    ----------
    tensor : Tensor
        the tensor to copy the shape from

    Returns
    -------
    Tensor
        the Inf filled tensor
    """

    return InfTensor(*tensor.shape, dtype=tensor.dtype, device=tensor.device)