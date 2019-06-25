import torch
from ..utility.LongTensor  import *
from ..utility.FloatTensor import *
from ..math.normvec        import *
from .subdivide            import *



def Icosahedron(device='cuda:0'):
    """
    Creates a single icosahedron mesh

    Parameters
    ----------
    device : str or torch.device (optional)
        the device the tensors will be stored to (default is 'cuda:0')

    Returns
    -------
    (Tensor,LongTensor,Tensor)
        the point set tensor, the topology tensor, the vertex normals
    """

    P = FloatTensor([[ 0       ,-0.525731, 0.850651],
                     [ 0.850651, 0       , 0.525731],
                     [ 0.850651, 0       ,-0.525731],
                     [-0.850651, 0       ,-0.525731],
                     [-0.850651, 0       , 0.525731],
                     [-0.525731, 0.850651, 0],
                     [ 0.525731, 0.850651, 0],
                     [ 0.525731,-0.850651, 0],
                     [-0.525731,-0.850651, 0],
                     [ 0       ,-0.525731,-0.850651],
                     [ 0       , 0.525731,-0.850651],
                     [ 0       , 0.525731, 0.850651]],device=device)
    T = torch.add(torch.t(LongTensor([\
        [ 2, 3, 7],[ 2, 8, 3],[ 4, 5, 6],[ 5, 4, 9],\
        [ 7, 6,12],[ 6, 7,11],[10,11, 3],[11,10, 4],\
        [ 8, 9,10],[ 9, 8, 1],[12, 1, 2],[ 1,12, 5],\
        [ 7, 3,11],[ 2, 7,12],[ 4, 6,11],[ 6, 5,12],\
        [ 3, 8,10],[ 8, 2, 1],[ 4,10, 9],[ 5, 9, 1]],device=device)),-1)
    N = normr(P.clone())
    return P,T,N



def Icosahedron_2(device='cuda:0'):
    """
    Creates a subdivided icosahedron mesh

    Parameters
    ----------
    device : str or torch.device (optional)
        the device the tensors will be stored to (default is 'cuda:0')

    Returns
    -------
    (Tensor,LongTensor,Tensor)
        the point set tensor, the topology tensor, the vertex normals
    """

    P,T = Icosahedron(device=device)[0:2]
    P,T = subdivide(P,T,1)
    P   = normr(P)
    N   = P.clone()
    return P,T,N



def Icosahedron_3(device='cuda:0'):
    """
    Creates a twice subdivided icosahedron mesh

    Parameters
    ----------
    device : str or torch.device (optional)
        the device the tensors will be stored to (default is 'cuda:0')

    Returns
    -------
    (Tensor,LongTensor,Tensor)
        the point set tensor, the topology tensor, the vertex normals
    """

    P,T = Icosahedron(device=device)[0:2]
    P,T = subdivide(P,T,2)
    P   = normr(P)
    N   = P.clone()
    return P,T,N
