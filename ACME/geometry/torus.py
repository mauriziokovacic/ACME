import torch
from ..utility.linspace   import *
from ..math.constant      import *
from ..math.cos           import *
from ..math.sin           import *
from ..topology.poly2poly import *
from .vertex_normal       import *
from .grid2mesh           import *


def TorusPatch(radius=(1, 0.5), tile=(20, 20), device='cuda:0'):
    """
    Creates a torus patch

    Parameters
    ----------
    radius : (float,float) (optional)
        radii of the torus (default is (1,0.5))
    tile : (int,int) (optional)
        the number of divisions of the cylinder (default is (20,20))
    device : str or torch.device (optional)
        the device the tensors will be stored to (default is 'cuda:0')

    Returns
    -------
    Tensor
        the (3,M,N,) tensor representing the torus
    """

    phi, theta = torch.meshgrid(linspace(0, PI2, tile[0], device=device),
                                linspace(0, PI2, tile[1], device=device))
    x = (radius[0] + radius[1] * cos(theta)) * cos(phi)
    y = (radius[0] + radius[1] * cos(theta)) * sin(phi)
    z = radius[1] * sin(theta)
    return torch.cat((x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0)), dim=0)


def Torus(radius=(1, 0.5), tile=(20, 20), device='cuda:0'):
    """
    Creates a torus quad mesh

    Parameters
    ----------
    radius : (float,float) (optional)
        radii of the torus (default is (1,0.5))
    tile : (int,int) (optional)
        the number of divisions of the cylinder (default is (20,20))
    device : str or torch.device (optional)
        the device the tensors will be stored to (default is 'cuda:0')

    Returns
    -------
    (Tensor,LongTensor,Tensor)
        the point set tensor, the topology tensor, the vertex normals
    """

    T, P = grid2mesh(*tuple(TorusPatch(radius=radius, tile=tile, device=device)))
    N    = vertex_normal(P, quad2tri(T))
    return P, T, N
