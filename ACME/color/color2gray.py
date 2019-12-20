from .color2float import *
from ..utility.strcmpi import *


def color2gray(C, dim=-1, method='linear'):
    """
    Converts the given color in grayscale using the specified method

    Parameters
    ----------
    C : Tensor
        the (3,W,H,) or (N,3,) RGB color tensor
    dim : int (optional)
        the dimension along the operation is performed (default is -1)
    method : str (optional)
        the conversion method. It should be one among 'linear', 'pal', 'hdtv', 'hdr' (default is 'linear')

    Returns
    -------
    Tensor
        the (W,H,) or (N,) grayscale color tensor

    Raises
    ------
    RuntimeError
        if method is not among the accepted ones
    """

    size      = list(C.shape)
    size[dim] = -1
    alpha     = torch.zeros(3, dtype=torch.float, device=C.device)
    if strcmpi(method, 'linear'):
        alpha[0] = 1 / 3
        alpha[1] = 1 / 3
        alpha[2] = 1 / 3
    elif strcmpi(method, 'hdtv'):
        alpha[0] = 0.2126
        alpha[1] = 0.7152
        alpha[2] = 0.0722
    elif strcmpi(method, 'pal'):
        alpha[0] = 0.299
        alpha[1] = 0.587
        alpha[2] = 0.114
    elif strcmpi(method, 'hdr'):
        alpha[0] = 0.2627
        alpha[1] = 0.6780
        alpha[2] = 0.0593
    else:
        raise RuntimeError('Unknown method.')
    return torch.sum(color2float(C) * alpha.expand(*size), dim=dim)
