import torch
from ACME.utility.linspace import *
from ACME.math.constant    import *
from ACME.math.cos         import *
from ACME.math.sin         import *
from .grid2mesh            import *


def Klein_Figure8(radius=3,res=50,device='cuda:0'):
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

    theta,nu = torch.meshgrid(linspace(0,PI2,res+1,device='cpu'),linspace(0,PI2,res+1,device='cpu'))
    x        = (r+cos(theta/2) * sin(nu) - sin(theta/2)*sin(2*nu)) * cos(theta)
    y        = (r+cos(theta/2) * sin(nu) - sin(theta/2)*sin(2*nu)) * sin(theta)
    z        = sin(theta/2) * sin(nu) + cos(theta/2)*sin(2*nu)
    T,P      = grid2mesh(x,y,z,device=device)
    return P,T
