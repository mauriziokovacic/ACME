import torch
from ..utility.linspace    import *
from ..utility.FloatTensor import *
from ..math.constant       import *
from ..math.cos            import *
from ..math.sin            import *
from ..math.normvec        import *
from .grid2mesh            import *



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

    n              = tile[0]+1
    m              = tile[1]+1
    theta          = torch.t(linspace(-1,1,n,device=device))*PI
    phi            = linspace(-1,1,m,device=device)*PI_2
    cosphi         = cos(phi)
    cosphi[ 0]     = 0
    cosphi[-1]     = 0
    sintheta       = sin(theta)
    sintheta[:, 0] = 0
    sintheta[:,-1] = 0
    x              = torch.mm(cosphi,cos(theta))
    y              = torch.mm(cosphi,sintheta)
    z              = torch.mm(sin(phi),torch.ones(1,n,dtype=torch.float,device=device))
    T,P            = grid2mesh(x,y,z)
    P              = normr(P)
    N              = P.clone()
    return P,T,N
