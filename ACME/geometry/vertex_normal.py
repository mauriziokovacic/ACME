import torch
from ..utility.accumarray import *
from ..utility.strcmpi    import *
from ..math.cross         import *
from ..math.normvec       import *
from .triangle_angle      import *

def vertex_normal(P,T,type=None):
    """
    Returns the normal of the vertices in the triangulation, averaged with the
    specified method.

    Parameters
    ----------
    P : Tensor
        the input point set
    T : LongTensor
        the topology tensor
    type : str (optional)
        the averaging method. Can be either None (mean), 'area' or 'angle' (default is None)

    Returns
    -------
    Tensor
        the vertex normals of the input mesh
    """

    Pi,Pj,Pk = P[T]
    Ni = cross(Pj-Pi,Pk-Pi)
    Nj = cross(Pk-Pj,Pi-Pj)
    Nk = cross(Pi-Pk,Pj-Pk)
    if type is not None:
        if strcmpi(type,'area'):
            Ai = torch.mul(norm(Ni),0.5)
            Aj = torch.mul(norm(Nj),0.5)
            Ak = torch.mul(norm(Nk),0.5)
            Ni = normr(Ni)*Ai
            Nj = normr(Nj)*Aj
            Nk = normr(Nk)*Ak
        if strcmpi(type,'angle'):
            Ai,Aj,Ak = triangle_angle(P,T)
            Ni = normr(Ni)*Ai
            Nj = normr(Nj)*Aj
            Nk = normr(Nk)*Ak
    I = torch.cat(tuple(T)).squeeze()
    V = torch.cat((Ni,Nj,Nk),dim=0)
    N = accumarray(I,V,size=row(P))
    return normr(N)
