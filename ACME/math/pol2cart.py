import torch
from ..utility.col import *
from .cos          import *
from .sin          import *


def pol2cart(P):
    """
    Converts input polar coordinates [theta,r[,z]] into cartesian coordinates [x,y[,z]]

    Parameters
    ----------
    P : Tensor
        the input (N,2,) or (N,3,) polar coordinates tensor

    Returns
    -------
    Tensor
        a (N,2,) or (N,3,) tensor containing [x,y[,z]]
    """

    theta, r = torch.t(P)[0:2]
    theta    = theta.unsqueeze(1)
    r        = r.unsqueeze(1)
    out      = torch.cat((r*cos(theta), r*sin(theta)), dim=1)
    if col(P) == 3:
        out = torch.cat((out, P[:, 2].unsqueeze(1)), dim=1)
    return out
