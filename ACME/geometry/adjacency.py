import torch
from ACME.utility.row        import *
from ACME.utility.col        import *
from ACME.utility.numel      import *
from ACME.utility.indices    import *
from ACME.utility.strcmpi    import *
from ACME.utility.to_row     import *
from ACME.utility.unique     import *
from ACME.topology.adjacency import *
from ACME.topology.poly2edge import *
from .triangle_cotangent     import *



def Adjacency(T,P=None,type='std',dtype=torch.float):
    """
    Creates the adjacency (combinatorial or with cotangent weights) matrix from the given input mesh

    Parameters
    ----------
    T : LongTensor
        the topology tensor
    P : Tensor (optional)
        the points set tensor (default is None)
    type : str (optional)
        the type of adjacency. It should be either 'std' or 'cot' (default is 'std')
    dtype : type (optional)
        the type of the output matrix (default is torch.float)

    Returns
    -------
    Tensor
        the adjacency matrix

    Raises
    ------
    AssertionError
        if type is not supported or if data is missing or badly shaped
    """

    if P is None:
        n = torch.max(T)[0]+1
    else:
        n = row(P)

    A = torch.zeros(n,n,dtype=dtype,device=T.device)

    if strcmpi(type,'std'):
        E = poly2edge(T)[0]
        E = torch.cat((E,flipud(E)),dim=1)
        return adjacency(E,torch.ones(col(E),dtype=dtype,device=T.device),size=(n,n))

    if strcmpi(type,'cot'):
        assert P is not None, 'Point tensor should not be None'
        assert istri(T), 'Cotangent adjacency defined only for triangular meshes'
        E = poly2edge(T)[0]
        W = 0.5*poly2edge(torch.cat(tuple(to_row(w) for w in triangle_cotangent(P,T)),dim=0))[0]
        return adjacency(E,W,size=(n,n))

    if type.lower()=='face':
        n   = col(T)
        E,t = poly2edge(T)
        I   = unique(torch.sort(torch.t(E),1)[0],ByRows=True)[2]
        E   = indices(0,col(E)-1,device=T.device)
        A   = torch.zeros(numel(E),n,dtype=torch.float,device=T.device)
        print(I.shape)
        for i,j in torch.cat((E[I],t),dim=1):
            A[i,j] = 1
        A   = torch.mm(torch.t(A),A)
        for i in range(0,row(A)):
            A[i,i] = 0
        return A

    assert False, 'Unknown output type'
