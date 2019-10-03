import torch


def quantize(tensor, levels):
    """
    Quantizes a given input tensor in range [0,1] into n levels

    Parameters
    ----------
    tensor : Tensor
        a tensor in range [0,1]
    levels : int
        the number of levels to quantize

    Returns
    -------
    Tensor
        the quantized tensor
    """

    return torch.div(torch.round(torch.mul(tensor, levels)), levels)
