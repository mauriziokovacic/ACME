import torch

def accumarray(I, V, size=None, default_value=0):
    """
    Returns a Tensor by accumulating elements of tensor V using the subscripts I

    The output tensor number of dimensions is/should be equal to the number of subscripts rows
    plus the values tensor number of dimensions minus one.

    Parameters
    ----------
    I : LongTensor
        the (N,) subscripts tensor
    V : Tensor
        the (M,F,) values tensor
    size : tuple (optional)
        the size of the output tensor. If None it will be automatically inferred (default is None)
    dim : int (optional)
        the dimension along the accumulation is performed (default is 0)
    default_value : float (optional)
        the default value of the output tensor (default is 0)

    Returns
    -------
    Tensor
        the accumulated tensor

    """

    if size is None:
        size      = list(V.size())
        size[0] = torch.max(I).item()+1
    return default_value + torch.zeros(size, dtype=V.dtype, device=V.device).scatter_add_(0, I.view(-1, 1).expand_as(V), V)
