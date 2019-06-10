import torch
from ACME.utility.FloatTensor import *
from .cos                     import *
from .sin                     import *
from .cross                   import *
from .normvec                 import *
from .eye                     import *
from .axang2rotm              import *
from .unitvec                 import *

def sph2rotm(S,scaling=False):
    """
    Creates a rotation matrix from a given spherical coordinate triplet [radius,theta,phi]

    In spherical coordinates radius represents the distance from the origin, theta is the
    elevation, and phi the azimuth

    Parameters
    ----------
    S : 1D Tensor
        a set of spherical coordinates in the shape of [radius,theta,phi]

    Returns
    -------
    Tensor
        a (3,3) rotation matrix
    """

    return torch.mm(eye(3)*(S[0] if scaling else 1),
                    torch.mm(axang2rotm(normr(cross(z_axis(),FloatTensor([[cos(S[1]),sin(S[1]),0]],device=S.device))).squeeze(), S[2]),
                             axang2rotm(z_axis().squeeze(), S[1])))
