import torch
from ACME.utility.linspace import *
from ACME.math.constant    import *
from ACME.math.cos         import *
from ACME.math.sin         import *
from .grid2mesh            import *

def Mobius_Strip(tile=(50,10),device='cuda:0'):
    """
    Creates a mobius strip quad mesh

    Parameters
    ----------
    tile : (int,int) (optional)
        the number of divisions of the strip (default is (50,10))
    device : str or torch.device (optional)
        the device the tensors will be stored to (default is 'cuda:0')

    Returns
    -------
    (Tensor,LongTensor)
        the point set tensor, the topology tensor
    """

    u   = linspace(0,PI2,tile[0]+1,device=device)
    v   = linspace(-0.5,0.5,tile[1]+1,device=device)
    u,v = torch.meshgrid(u,v)
    x   = (1+v*cos(u/2))*cos(u)
    y   = (1+v*cos(u/2))*sin(u)
    z   = v*sin(u/2)
    T,P = grid2mesh(x,y,z)
    return P,T
