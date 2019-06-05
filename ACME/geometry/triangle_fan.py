import torch
from utility.row         import *
from utility.repmat      import *
from utility.LongTensor  import *
from utility.FloatTensor import *
from .equilater_polygon  import *

def TriangleFan(n,device='cuda:0'):
    """
    Creates a n-triangle fan

    Parameters
    ----------
    n : int
        the number of triangles in the fan
    device : str or torch.device (optional)
        the device the tensors will be stored to (default is 'cuda:0')

    Returns
    -------
    (Tensor,LongTensor,Tensor)
        the point set tensor, the topology tensor, the vertex normals
    """

    P = equilaterla_polygon(n,device)
    P = torch.cat((torch.zeros(1,3,dtype=torch.float,device=device),P),dim=0)
    N = repmat(FloatTensor([[0,0,1]],device=device),(row(P),1))
    T = torch.cat((torch.zeros(n,1,dtype=torch.long,device=device),
                   torch.reshape(indices(2,row(P)-1,device=device),(n,1)),
                   torch.reshape(indices(3,row(P)  ,device=device),(n,1))),dim=1)
    return P,T,N
