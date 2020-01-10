from .color2float import *


def color2gray(C, weight=(1/3, 1/3, 1/3), dim=-1):
    """
    Converts the given color in grayscale using the specified weights

    Parameters
    ----------
    C : Tensor
        the (3,W,H,) or (N,3,) RGB color tensor
    weight : list or tuple (optional)
        the weighting factors for generating the grayscale image (default is (1/3, 1/3, 1/3))
    dim : int (optional)
        the dimension along the operation is performed (default is -1)

    Returns
    -------
    Tensor
        the (W,H,) or (N,) grayscale color tensor
    """

    size = list(C.shape)
    size[dim] = -1
    alpha = torch.tensor(weight, dtype=torch.float, device=C.device)
    return torch.sum(color2float(C) * alpha.expand(*size), dim=dim)


def color2gray_pal(C, dim=-1):
    """
    Converts the given color in grayscale with PAL standard

    Parameters
    ----------
    C : Tensor
        the (3,W,H,) or (N,3,) RGB color tensor
    dim : int (optional)
        the dimension along the operation is performed (default is -1)

    Returns
    -------
    Tensor
        the (W,H,) or (N,) grayscale color tensor
    """

    return color2gray(C, weight=[0.299, 0.587, 0.114], dim=dim)


def color2gray_hdtv(C, dim=-1):
    """
    Converts the given color in grayscale with HDTV standard

    Parameters
    ----------
    C : Tensor
        the (3,W,H,) or (N,3,) RGB color tensor
    dim : int (optional)
        the dimension along the operation is performed (default is -1)

    Returns
    -------
    Tensor
        the (W,H,) or (N,) grayscale color tensor
    """

    return color2gray(C, weight=[0.2126, 0.7152, 0.0722], dim=-1)


def color2gray_hdr(C, dim=-1):
    """
    Converts the given color in grayscale with HDR standard

    Parameters
    ----------
    C : Tensor
        the (3,W,H,) or (N,3,) RGB color tensor
    dim : int (optional)
        the dimension along the operation is performed (default is -1)

    Returns
    -------
    Tensor
        the (W,H,) or (N,) grayscale color tensor
    """

    return color2gray(C, weight=[0.2627, 0.6780, 0.0593], dim=dim)
