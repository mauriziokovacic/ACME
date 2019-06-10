import torch
from ACME.utility.row   import *
from ACME.math.constant import *

def sph2color(S,rmax=None):
    """
    Converts the given spherical coordinates tensor into a color

    Parameters
    ----------
    S : Tensor
        the spherical coordinate [r,theta,phi] tensor
    rmax : None (optional)
        the maximum radius. If None it will be fixed to 0 (default is None)

    Returns
    -------
    Tensor
        the color tensor
    """

    r,theta,phi = torch.t(S)
    if rmax is None:
        r = torch.mul(r,0)
    else:
        r = torch.div(r,rmax)
    theta = torch.div(torch.add(theta,PI),PI2)
    phi   = torch.div(torch.add(phi,PI),PI2)
    r     = torch.reshape(r,(row(S),1))
    theta = torch.reshape(theta,(row(S),1))
    phi   = torch.reshape(phi,(row(S),1))
    return torch.cat((r,theta,phi),dim=1)
