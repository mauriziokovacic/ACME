import torch
from color.color2float import *

def color2normal(C):
    """
    Converts a given color into a normal direction

    Parameters
    ----------
    C : Tensor
        a color (n,3) tensor

    Returns
    -------
    Tensor
        a normal tensor
    """

    return torch.add(torch.mul(color2float(C),2),-1)
