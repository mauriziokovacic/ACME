import torch
from utility.row         import *
from utility.col         import *
from utility.numel       import *
from utility.strcmpi     import *
from utility.to_column   import *
from .triangle_cotangent import *
from topology.poly2ind   import *
from topology.poly2poly  import *



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
        for i,j in torch.t(E):
            A[i,j] = 1
        return A

    if strcmpi(type,'cot'):
        assert P is not None, 'Point tensor should not be None'
        assert istri(T), 'Cotangent adjacency defined only for triangular meshes'
        I,J,K = tri2ind(T)
        CTi,CTj,CTk = triangle_cotangent(P,T)
        W = torch.cat((to_column(torch.cat((I,J,K,J,K,I))).to(dtype=dtype),\
                       to_column(torch.cat((J,K,I,I,J,K))).to(dtype=dtype),\
                       to_column(torch.mul(torch.cat((CTk,CTi,CTj,CTk,CTi,CTj),dim=0),0.5))),dim=1)
        for i,j,w in W:
            i = i.to(dtype=torch.long)
            j = j.to(dtype=torch.long)
            A[i,j] += w
        return A

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
