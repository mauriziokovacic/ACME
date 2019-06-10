import torch
from ACME.utility.row      import *
from ACME.utility.linspace import *
from ACME.math.constant    import *
from ACME.math.cos         import *
from ACME.math.sin         import *
from ACME.math.normvec     import *
from .grid2mesh            import *

def Cylinder(tile=(16,8),radius=1,device='cuda:0'):
    """
    Creates a cylinder quad mesh

    Parameters
    ----------
    tile : (int,int) (optional)
        the number of divisions of the cylinder (default is (16,8))
    radius : float (optional)
        radius of the cylinder (default is 1)
    device : str or torch.device (optional)
        the device the tensors will be stored to (default is 'cuda:0')

    Returns
    -------
    (Tensor,LongTensor,Tensor)
        the point set tensor, the topology tensor, the vertex normals
    """

    x,y,z = torch.meshgrid(torch.mul(cos(linspace(0,PI2,tile[0]+1,device='cpu')),radius),\
                           torch.mul(sin(linspace(0,PI2,tile[0]+1,device='cpu')),radius),\
                           linspace(-1,1,tile[1]+1,device='cpu'))
    T,P   = grid2mesh(x,y,z,device=device)
    N     = torch.cat((normr(P[:,0:2]),torch.zeros(row(P),1,dtype=torch.float,device=device)),dim=1)
    return P,T,N
