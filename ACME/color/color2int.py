import torch


def color2int(C):
    """
    Converts a given color tensor in [0,1] float representation to [0,255] uint8

    Parameters
    ----------
    C : Tensor
        a torch.float (n,3) tensor

    Returns
    -------
    Tensor
        a torch.uint8 (n,3) tensor
    """

    c = C.clone()
    if (C.dtype != torch.uint8) or (torch.max(c).item() <= 1):
        c = torch.clamp(torch.round(torch.mul(C, 255)), min=0, max=255).to(dtype=torch.uint8)
    return c
