import torch


def quantize(tensor, levels, dtype=torch.float):
    """
    Quantizes a given input tensor in range [0,1] into n levels.

    Parameters
    ----------
    tensor : Tensor
        a tensor in range [0,1]
    levels : int
        the number of levels to quantize
    dtype : torch.dtype
        the output tensor type. If 'torch.long' is used, the output tensor has entries in [0, levels].

    Returns
    -------
    Tensor or LongTensor
        the quantized tensor
    """

    if dtype == torch.float:
        return torch.div(torch.round(torch.mul(tensor, levels)), levels)
    return torch.round(torch.mul(tensor, levels)).to(dtype=torch.long)
