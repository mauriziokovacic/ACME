import torch


def demean(tensor, dim=-1):
    """
    Demeans the input tensor along the specified dimension(s)

    Parameters
    ----------
    tensor : Tensor
        the input tensor
    dim : int or list or tuple (optional)
        the dimension(s) along the operation is performed (default is -1)

    Returns
    -------
    Tensor
        the demeaned tensor
    """

    return tensor - torch.mean(tensor, dim=dim, keepdim=True)
