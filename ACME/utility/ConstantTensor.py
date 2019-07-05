import torch


def ConstantTensor(value, *size, dtype=torch.float, device='cuda:0'):
    """
    Returns a Tensor containing only value

    Parameters
    ----------
    value : int or float
        the value of every item in the Tensor
    *size : int...
        the shape of the tensor
    dtype : type (optional)
        the type of the Tensor (default is torch.float)
    device : str
        the device to store the tensor to

    Returns
    -------
    Tensor
        a tensor made out of occurrencies of the input value
    """

    return torch.empty(*size, dtype=dtype, device=device).fill_(value)


