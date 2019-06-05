import torch

def normalize(tensor,min=None,max=None):
    """
    Normalize the tensor values between [0-1]

    Parameters
    ----------
    tensor : Tensor
        the input tensor
    min : int or float (optional)
        the value to be considered zero. If None, min(tensor) will be used instead (default is None)
    max : int or float (optional)
        the value to be considered one. If None, max(tensor) will be used instead (default is None)

    Returns
    -------
    Tensor
        a tensor with [min-max] retargeted in [0-1]
    """

    if min is None:
        min = torch.min(tensor)
    if max is None:
        max = torch.max(tensor)
    return torch.div(torch.add(tensor,-min),(max-min))
