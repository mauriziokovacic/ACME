import torch
from ACME.utility.linspace    import *
from ACME.utility.repmat      import *
from ACME.utility.FloatTensor import *
from .grid2mesh               import *

def Plane(tile=(2,2),device='cuda:0'):
    """
    Creates a gridded plane quad mesh

    Parameters
    ----------
    tile : (int,int) (optional)
        the number of tiles in the plane (default is (2,2))
    device : str or torch.device (optional)
        the device the tensors will be stored to (default is 'cuda:0')

    Returns
    -------
    (Tensor,LongTensor,Tensor)
        the point set tensor, the topology tensor, the vertex normals
    """

    x,y = torch.meshgrid(linspace(-1/2,1/2,tile[0]+1,device='cpu'),\
                         linspace(-1/2,1/2,tile[1]+1,device='cpu'))
    T,P = grid2mesh(x,y,torch.zeros_like(x),device=device)
    N   = repmat(FloatTensor([[0,0,1]],device=device),(row(P),1))
    return P,T,N
