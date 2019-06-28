import torch

def color2float(C):
    """
    Converts a given color tensor in [0,255] uint8 representation to [0,1] float

    Parameters
    ----------
    C : Tensor
        a torch.uint8 (n,3) tensor

    Returns
    -------
    Tensor
        a torch.float (n,3) tensor
    """

    c = C
    if (C.dtype != torch.float) or (torch.max(c).item()>1):
        c = torch.clamp(torch.div(C.to(dtype=torch.float),255),min=0,max=1)
    return c
