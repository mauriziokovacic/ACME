import torch
from utility.col import *
from .norm import *

def cart2pol(P):
    """
    Converts the input cartesian coordinates [x,y[,z]] into polar coordinates [theta,r[,z]]

    Parameters
    ----------
    P : Tensor
        a nx2 or nx3 tensor representing 2D or 3D points

    Returns
    -------
    Tensor
        a nx2 or nx3 tensor containing [theta,r] or [theta,r,z]
    """

    x,y   = torch.t(P)[0:2]
    x     = x.unsqueeze(1)
    y     = y.unsqueeze(1)
    theta = torch.atan2(y,x)
    r     = hypot(x,y)
    out   = torch.cat((theta,r),dim=1)
    if col(P)==3:
        out = torch.cat((out,P[:,2].unsqueeze(1)),dim=1)
    return out
