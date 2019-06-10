import torch
from ACME.utility.linspace    import *
from ACME.utility.FloatTensor import *
from ACME.utility.repmat      import *
from ACME.math.constant       import *
from ACME.math.pol2cart       import *
from .grid2mesh               import *



def Disk(tile=(8,8),device='cuda:0'):
    """
    Creates a disk quad mesh

    Parameters
    ----------
    tile : (int,int) (optional)
        the number of divisions of the disk (default is (8,8))
    device : str or torch.device (optional)
        the device the tensors will be stored to (default is 'cuda:0')

    Returns
    -------
    (Tensor,LongTensor,Tensor)
        the point set tensor, the topology tensor, the vertex normals
    """

    r   = torch.reshape(linspace(0,1,tile[0]+1,device='cpu'),(1,tile[0]+1))
    t   = torch.reshape(linspace(0,PI2,tile[1]+1,device='cpu'),(tile[1]+1,1))
    r   = repmat(r,(tile[1]+1,1))
    t   = repmat(t,(1,tile[0]+1))
    T,P = grid2mesh(t,r,torch.zeros_like(x),device=device)
    P   = pol2cart(P)
    N   = repmat(FloatTensor([[0,0,1]],device=device),(row(P),1))
    return P,T,N
