import torch
from ..utility.LongTensor  import *
from ..utility.FloatTensor import *
from ..math.constant       import *
from ..topology.polyflip   import *
from .vertex_normal        import *


def Tetrahedron(device='cuda:0'):
    """
    Creates a single tetrahedron mesh

    Parameters
    ----------
    device : str or torch.device (optional)
        the device the tensors will be stored to (default is 'cuda:0')

    Returns
    -------
    (Tensor,LongTensor,Tensor)
        the point set tensor, the topology tensor, the vertex normals
    """

    P = FloatTensor([[ 1/3*SQRT3,   0,           0],
                     [-1/6*SQRT3, 1/2,           0],
                     [-1/6*SQRT3,-1/2,           0],
                     [           0,   0, 1/3*SQRT6]], device=device)
    T = polyflip(torch.add(torch.t(LongTensor([[1,2,3],[1,3,4],[1,4,2],[3,2,4]], device=device)), -1))
    N = vertex_normal(P, T)
    return P, T, N
