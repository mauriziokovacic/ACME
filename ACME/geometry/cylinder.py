import torch
from ..utility.row      import *
from ..utility.linspace import *
from ..math.constant    import *
from ..math.cos         import *
from ..math.sin         import *
from ..math.normvec     import *
from .grid2mesh         import *


def CylinderPatch(tile=(16, 8), radius=1, device='cuda:0'):
    """
    Creates a cylinder patch

    Parameters
    ----------
    tile : (int,int) (optional)
        the number of divisions of the cylinder (default is (16,8))
    radius : float (optional)
        radius of the cylinder (default is 1)
    device : str or torch.device (optional)
        the device the tensors will be stored to (default is 'cuda:0')

    Returns
    -------
    Tensor
        the (3,M,N,) tensor representing the cylinder
    """

    x, y, z = torch.meshgrid(torch.mul(cos(linspace(0, PI2, tile[0] + 1, device=device)), radius),
                             torch.mul(sin(linspace(0, PI2, tile[0] + 1, device=device)), radius),
                             linspace(-1, 1, tile[1] + 1, device='cpu'))
    return torch.cat((x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0)), dim=0)


def Cylinder(tile=(16, 8), radius=1, device='cuda:0'):
    """
    Creates a cylinder quad mesh

    Parameters
    ----------
    tile : (int,int) (optional)
        the number of divisions of the cylinder (default is (16,8))
    radius : float (optional)
        radius of the cylinder (default is 1)
    device : str or torch.device (optional)
        the device the tensors will be stored to (default is 'cuda:0')

    Returns
    -------
    (Tensor,LongTensor,Tensor)
        the point set tensor, the topology tensor, the vertex normals
    """

    T, P = grid2mesh(*tuple(CylinderPatch(tile=tile, radius=radius, device=device)))
    N    = torch.cat((normr(P[:, 0:2]), torch.zeros(row(P), 1, dtype=torch.float, device=device)), dim=1)
    return P, T, N
