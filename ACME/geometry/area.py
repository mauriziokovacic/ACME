import torch
from ACME.utility.row        import *
from ACME.utility.accumarray import *
from ACME.utility.repmat     import *
from ACME.utility.to_column  import *
from ACME.math.cross         import *
from ACME.math.norm          import *

def triangle_area(P,T):
    """
    Returns the triangle area of all the input triangles

    Parameters
    ----------
    P : Tensor
        a (n,3) point set tensor
    T : LongTensor
        the topology tensor

    Returns
    -------
    Tensor
        the triangle areas
    """

    Pi,Pj,Pk = P[T]
    return torch.mul(norm(cross(Pj-Pi,Pk-Pi,dim=1),dim=1),0.5)



def barycentric_area(P,T):
    """
    Returns the barycentric area of each vertex in the triangulation

    Parameters
    ----------
    P : Tensor
        the input point set tensor
    T : LongTensor
        the topology tensor

    Returns
    -------
    Tensor
        the vertex barycentric area
    """

    A     = triangle_area(P,T)/3
    return to_column(accumarray(torch.cat(tuple(T),dim=0).squeeze(),repmat(A,3,1).squeeze(),row(P)))
