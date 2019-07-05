import torch
from ..utility.LongTensor   import *
from ..utility.FloatTensor  import *
from ..topology.subdivision import *
from ..math.normvec         import *
from .subdivide             import *


def Octahedron(device='cuda:0'):
    """
    Creates a single octahedron mesh

    Parameters
    ----------
    device : str or torch.device (optional)
        the device the tensors will be stored to (default is 'cuda:0')

    Returns
    -------
    (Tensor,LongTensor,Tensor)
        the point set tensor, the topology tensor, the vertex normals
    """

    P = FloatTensor([[0,0,1],[1,0,0],[0,1,0],[-1,0,0],[0,-1,0],[0,0,-1]], device=device)
    N = P.clone()
    T = torch.add(torch.t(LongTensor([[1,2,3],[1,3,4],[1,4,5],[1,5,2],[6,3,2],[6,4,3],[6,5,4],[6,2,5]], device=device)), -1)
    return P, T, N


def Octahedron_2(device='cuda:0'):
    """
    Creates a subdivided octahedron mesh

    Parameters
    ----------
    device : str or torch.device (optional)
        the device the tensors will be stored to (default is 'cuda:0')

    Returns
    -------
    (Tensor,LongTensor,Tensor)
        the point set tensor, the topology tensor, the vertex normals
    """

    P, T = Octahedron(device=device)[0:2]
    P, T = subdivide(P, T, 1)[0:2]
    P    = normr(P)
    N    = P.clone()
    return P, T, N



def Octahedron_3(device='cuda:0'):
    """
    Creates a twice subdivided octahedron mesh

    Parameters
    ----------
    device : str or torch.device (optional)
        the device the tensors will be stored to (default is 'cuda:0')

    Returns
    -------
    (Tensor,LongTensor,Tensor)
        the point set tensor, the topology tensor, the vertex normals
    """

    P, T = Octahedron(device=device)[0:2]
    P, T = subdivide(P, T, 2)[0:2]
    P    = normr(P)
    N    = P.clone()
    return P, T, N
