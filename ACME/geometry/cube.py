import torch
from ACME.utility.LongTensor  import *
from ACME.utility.FloatTensor import *
from ACME.math.normvec        import *
from ACME.topology.polyflip   import *



def Cube(device='cuda:0'):
    """
    Creates a single cube mesh

    Parameters
    ----------
    device : str or torch.device (optional)
        the device the tensors will be stored to (default is 'cuda:0')

    Returns
    -------
    (Tensor,LongTensor,Tensor)
        the point set tensor, the topology tensor, the vertex normals
    """

    P = FloatTensor([[+1,+1,+1],[-1,+1,+1],[+1,-1,+1],[-1,-1,+1],\
                     [+1,+1,-1],[-1,+1,-1],[+1,-1,-1],[-1,-1,-1]],device=device)
    T = torch.t(LongTensor([[0,1,3,2],[5,4,6,7],[0,4,5,1],[2,3,7,6],[0,2,6,4],[7,3,1,5]],device=device))
    N = normr(P)
    return P,T,N
