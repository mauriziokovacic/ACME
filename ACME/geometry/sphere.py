import torch
from ACME.utility.linspace    import *
from ACME.utility.FloatTensor import *
from ACME.math.constant       import *
from ACME.math.cos            import *
from ACME.math.sin            import *
from ACME.math.normvec        import *
from .grid2mesh               import *

def Sphere(n=20,device='cuda:0'):
    """
    Creates a unit sphere quad mesh

    Parameters
    ----------
    n : int (optional)
        the resolution of the sphere (default is 20)
    device : str or torch.device (optional)
        the device the tensors will be stored to (default is 'cuda:0')

    Returns
    -------
    (Tensor,LongTensor,Tensor)
        the point set tensor, the topology tensor, the vertex normals
    """

    theta        = FloatTensor(list(range(-n,n,2)),device='cpu').unsqueeze(0)/n*PI
    phi          = FloatTensor(list(range(-n,n,2)),device='cpu').unsqueeze(1)/n*PI_2
    cosphi       = cos(phi)
    cosphi[ 0]   = 0
    cosphi[-1]   = 0
    sintheta     = sin(theta)
    sintheta[ 0] = 0
    sintheta[-1] = 0
    x            = torch.mm(cosphi,cos(theta))
    y            = torch.mm(cosphi,sintheta)
    z            = torch.mm(sin(phi),torch.ones(1,n,dtype=torch.float))
    T,P          = grid2mesh(x,y,z,device=device)
    N            = normr(P)
    return P,T,N
