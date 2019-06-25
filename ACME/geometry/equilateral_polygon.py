import torch
from ..utility.linspace import *
from ..math.cos         import *
from ..math.sin         import *

def equilateral_polygon(n,device='cuda:0'):
    """
    Creates the vertices of an equilateral n-gon

    Parameters
    ----------
    n : int
        the number of vertices of the n-gon
    device : str or torch.device (optional)
        the device the tensors will be stored to (default is 'cuda:0')

    Returns
    -------
    Tensor
        the vertices of the n-gon
    """

    t = linspace(0,PI2,n+1,device=device)
    t = t[0:n].unsqueeze(1)
    return torch.cat((cos(t),sin(t),torch.zeros_like(t,device=device)),dim=1)
