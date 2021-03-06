import torch
from ..utility.LongTensor  import *
from ..utility.FloatTensor import *
from ..math.constant       import *
from ..math.normvec        import *
from ..topology.polyflip   import *


def Hexahedron(device='cuda:0'):
    """
    Creates a single hexahedron mesh

    Parameters
    ----------
    device : str or torch.device (optional)
        the device the tensors will be stored to (default is 'cuda:0')

    Returns
    -------
    (Tensor,LongTensor,Tensor)
        the point set tensor, the topology tensor, the vertex normals
    """

    P = FloatTensor([[ 1/3*SQRT3,   0,         0],
                     [-1/6*SQRT3, 1/2,         0],
                     [-1/6*SQRT3,-1/2,         0],
                     [         0,   0, 1/3*SQRT6],
                     [         0,   0,-1/3*SQRT6]], device=device)
    P = P-torch.mean(P, 1)
    T = torch.t(torch.cat((polyflip(LongTensor([[1, 3, 4], [1, 4, 2], [3, 2, 4]], device=device)),
                           LongTensor([[1, 3, 5], [1, 5, 2], [3, 2, 5]], device=device)), dim=0))
    N = normr(P.clone())
    return P, T, N
