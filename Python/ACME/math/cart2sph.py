import torch
from .norm import *

def cart2sph(P):
    """
    Converts the input cartesian coordinates [x,y,z] into spherical coordinates [r,theta,phi]

    In spherical coordinates, r represents the radius, theta the elevetion and phi the azimuth

    Parameters
    ----------
    P : Tensor
        a nx3 tensor containing [x,y,z]

    Returns
    -------
    Tensor
        a nx3 tensor containing [r,theta,phi]
    """

    x,y,z = torch.t(P)
    x     = x.unsqueeze(1)
    y     = y.unsqueeze(1)
    z     = z.unsqueeze(1)
    r     = pnorm(P)                  #radius
    theta = torch.atan2(z,hypot(x,y)) #elevation
    phi   = torch.atan2(y,x)          #azimuth
    return torch.cat((r,theta,phi),dim=1)
