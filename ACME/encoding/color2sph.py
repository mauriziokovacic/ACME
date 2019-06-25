import torch
from ..math.constant     import *
from ..color.color2float import *

def color2sph(C,rmax=None):
    """
    Converts a given color tensor into spherical coordinates

    Parameters
    ----------
    C : Tensor
        the color tensor
    rmax : float (optional)
        the maximum radius. If None it will be set to 1 (default is None)

    Returns
    -------
    Tensor
        the spherical coordinates [r,theta,phi] tensor
    """

    r,g,b = torch.t(color2float(C))
    r = torch.reshape(r,(row(C),1))
    g = torch.reshape(g,(row(C),1))
    b = torch.reshape(b,(row(C),1))
    if rmax is not None:
        r = torch.mul(r,rmax)
    else:
        r = torch.ones(1,1,dtype=torch.float,device=C.device)
    return torch.cat((r,torch.mul(g,PI2),torch.mul(b,PI2)),dim=1)
