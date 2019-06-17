import torch
from ACME.utility.linspace    import *
from ACME.utility.FloatTensor import *
from ACME.math.constant       import *
from ACME.math.cos            import *
from ACME.math.sin            import *
from ACME.math.normvec        import *
from .grid2mesh               import *



def Sphere(tile=(20,20),device='cuda:0'):
    """
    Creates a unit sphere quad mesh

    Parameters
    ----------
    tile : tuple (optional)
        the resolution of the sphere along azimuth end elevation (default is (20,20))
    device : str or torch.device (optional)
        the device the tensors will be stored to (default is 'cuda:0')

    Returns
    -------
    (Tensor,LongTensor,Tensor)
        the point set tensor, the topology tensor, the vertex normals
    """

    n            = tile[0]
    m            = tile[1]
    theta        = FloatTensor(list(range(-n,n,2)),device='cpu').unsqueeze(0)/n*PI
    phi          = FloatTensor(list(range(-m,m,2)),device='cpu').unsqueeze(1)/m*PI_2
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
    N            = normr(P.clone())
    return P,T,N
