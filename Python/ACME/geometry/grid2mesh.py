import torch
from utility.ismatrix import *
from utility.row      import *
from utility.col      import *
from utility.indices  import *
from utility.su2ind   import *

def grid2mesh(X,Y,Z=None,device='cuda:0'):
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
    device : str or torch.device
        the device the tensors will be stored to (default is 'cuda:0')

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
    n = row(X)
    m = col(X)
    P = torch.cat((torch.reshape(X,(n*m,1)),torch.reshape(Y,(n*m,1))),dim=1)
    if Z is not None:
        assert ismatrix(Z), "Z tensor must be a matrix."
        assert row(Z)>=2 and col(Z)>=2, "Z tensor must be at least 2x2."
        assert X.shape==Z.shape, "Tensors must have same shape. Got X={} and Z={}".format(X.shape,Z.shape)
        P = torch.cat((P,torch.reshape(Z,(n*m,1))),dim=1)
    s = X.shape
    T = torch.t(torch.cat((sub2ind(s,indices(0,m-1,device=device),indices(0,n-1,device=device)),\
                           sub2ind(s,indices(1,m  ,device=device),indices(0,n-1,device=device)),\
                           sub2ind(s,indices(1,m  ,device=device),indices(1,n  ,device=device)),\
                           sub2ind(s,indices(0,m-1,device=device),indices(1,n  ,device=device))),dim=1))
    P = P.to(device=device)
    return T,P
