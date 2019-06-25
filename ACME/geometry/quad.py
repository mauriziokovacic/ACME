import torch
from ..utility.row         import *
from ..utility.repmat      import *
from ..utility.LongTensor  import *
from ..utility.FloatTensor import *

def Quad(device='cuda:0'):
    """
    Creates a single quad mesh

    Parameters
    ----------
    device : str or torch.device (optional)
        the device the tensors will be stored to (default is 'cuda:0')

    Returns
    -------
    (Tensor,LongTensor,Tensor)
        the point set tensor, the topology tensor, the vertex normals
    """

    P = FloatTensor([[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]],device=device)
    T = torch.t(LongTensor([[0,1,2,3]],device=device))
    N = repmat(FloatTensor([[0,0,1]],device=device),(row(P),1))
    return P,T,N
