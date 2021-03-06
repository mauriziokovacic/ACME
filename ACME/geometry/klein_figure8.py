import torch
from ..utility.linspace import *
from ..math.constant    import *
from ..math.cos         import *
from ..math.sin         import *
from .grid2mesh         import *


def Klein_Figure8(radius=3, res=50, device='cuda:0'):
    """
    Creates a Klein bottle with figure 8 quad mesh

    Parameters
    ----------
    radius : float (optional)
        the Klein bottle radius (default is 3)
    res : int (optional)
        the resolution of the output mesh (default is 50)
    device : str or torch.device (optional)
        the device the tensors will be stored to (default is 'cuda:0')

    Returns
    -------
    (Tensor,LongTensor)
        the point set tensor, the topology tensor
    """

    theta,nu = torch.meshgrid(linspace(0, PI2, res+1, device=device),
                              linspace(0, PI2, res+1, device=device))
    x        = (radius+cos(theta/2) * sin(nu) - sin(theta/2)*sin(2*nu)) * cos(theta)
    y        = (radius+cos(theta/2) * sin(nu) - sin(theta/2)*sin(2*nu)) * sin(theta)
    z        = sin(theta/2) * sin(nu) + cos(theta/2)*sin(2*nu)
    T,P      = grid2mesh(x, y, z)
    return P, T
