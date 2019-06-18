import torch
from ACME.utility.ismatrix import *
from ACME.utility.row      import *
from ACME.utility.col      import *
from ACME.utility.indices  import *

def grid2mesh(X,Y,Z=None):
    """
    Creates a 2D or 3D quad mesh from a given set of coordinates in matrix form

    Parameters
    ----------
    X : Tensor
        a (n,m) tensor representing the x coordinate
    Y : Tensor
        a (n,m) tensor representing the y coordinate
    Z : Tensor (optional)
        a (n,m) tensor representing the z coordinate (default is None)

    Returns
    -------
    (LongTensor,Tensor)
        the topology tensor and the points set

    Raises
    ------
    AssertionError
        if X,Y (and Z) are not matrix with identical shape
    """

    assert ismatrix(X), "X tensor must be a matrix."
    assert row(X)>=2 and col(X)>=2, "X tensor must be at least 2x2."
    assert ismatrix(Y), "Y tensor must be a matrix."
    assert row(Y)>=2 and col(Y)>=2, "Y tensor must be at least 2x2."
    assert X.shape==Y.shape, "Tensors must have same shape. Got X={} and Y={}".format(X.shape,Y.shape)
    m = row(X)
    n = col(X)
    P = torch.cat((torch.t(X).flatten().unsqueeze(1),torch.t(Y).flatten().unsqueeze(1)),dim=1)
    if Z is not None:
        assert ismatrix(Z), "Z tensor must be a matrix."
        assert row(Z)>=2 and col(Z)>=2, "Z tensor must be at least 2x2."
        assert X.shape==Z.shape, "Tensors must have same shape. Got X={} and Z={}".format(X.shape,Z.shape)
        P = torch.cat((P,torch.t(Z).flatten().unsqueeze(1)),dim=1)
    q = indices(1,m*n-m-1,device=X.device)
    q = q[(q%m)!=0].unsqueeze(1)
    T = torch.t(torch.cat((q,q+m,q+m+1,q+1),dim=1))-1
    return T,P
