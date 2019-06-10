import numpy
import torch
from ACME.utility.flatten     import *
from ACME.utility.transpose   import *
from ACME.utility.numpy2torch import *
from ACME.utility.torch2numpy import *

def __xmesh(T,scheme,iter=None):
    """
    Creates the subdivision matrix of the given topology tensor w.r.t. the input scheme.

    Parameters
    ----------
    T : LongTensor
        the topology tensor
    scheme : (numpy.ndarray,int)
        the subdivision scheme with the resulting polygons sides
    iter : int (optional)
        the number of subdivision iteration to perform. No iteration if None (default is None)

    Returns
    -------
    (Tensor,LongTensor)
        the subdivision matrix and the new topology tensor
    """

    def repel(a,dim):
      out = a
      for d in range(0,len(dim)):
        out = numpy.repeat(out,dim[d],axis=d)
      return out
    t = torch.t(T).cpu().numpy()
    M = numpy.identity(numpy.max(t)+1)
    if iter is not None:
        t = flatten(t)
        M = M[t]
        for i in range(0,iter):
            e = numpy.tile(scheme[0],(row(M)//row(T),1)) + repel(torch2numpy(indices(0,row(M)//row(T)-1,device='cpu')),(row(scheme[0]),2))*row(T)
            m = numpy.zeros((row(e),col(M)))
            for j in range(0,col(e)):
                m = m+M[e[:,j]]
            M = (1/col(e))*m
        t = transpose(numpy.reshape(torch2numpy(indices(0,row(M)-1,device='cpu')),(int(row(M)/3),3)))
    return numpy2torch(M,dtype=torch.float,device=T.device),\
           numpy2torch(t,dtype=torch.long,device=T.device)



def xtri(T,iter=None):
    """
    Creates the naive subdivision matrix for a triangle topology.

    Parameters
    ----------
    T : LongTensor
        the topology tensor
    iter : int (optional)
        the number of iteration to perform. No iteration if None (default is None)

    Returns
    -------
    (Tensor,LongTensor)
        the subdivision matrix and the new topology tensor
    """

    i = 0
    j = 1
    k = 2
    scheme = (numpy.array([[i,i],[i,j],[i,k],
                           [j,j],[j,k],[j,i],
                           [k,k],[k,i],[k,j],
                           [i,j],[j,k],[k,i]]),3)
    return __xmesh(T,scheme,iter=iter)



def xquad(T,iter=None):
    """
    Creates the naive subdivision matrix for a quad topology.

    Parameters
    ----------
    T : LongTensor
        the topology tensor
    iter : int (optional)
        the number of iteration to perform. No iteration if None (default is None)

    Returns
    -------
    (Tensor,LongTensor)
        the subdivision matrix and the new topology tensor
    """

    i = 0
    j = 1
    k = 2
    l = 3
    scheme = (numpy.array([[i,i,i,i],[i,i,j,j],[i,j,k,l],[i,i,l,l],
                           [j,j,j,j],[j,j,k,k],[i,j,k,l],[j,j,i,i],
                           [k,k,k,k],[k,k,l,l],[i,j,k,l],[k,k,j,j],
                           [l,l,l,l],[l,l,i,i],[i,j,k,l],[l,l,k,k]]),4)
    return __xmesh(T,scheme,iter=iter)



def xtri2quad(T):
    """
    Creates the subdivision matrix for transforming a triangle topology into a quad one

    Parameters
    ----------
    T : LongTensor
        the topology tensor

    Returns
    -------
    (Tensor,LongTensor)
        the subdivision matrix and the new topology tensor
    """

    i = 0
    j = 1
    k = 2
    scheme = (numpy.array([[i,i,i,i,i,i],[i,i,i,j,j,j],[i,i,j,j,k,k],[i,i,i,k,k,k],
                           [j,j,j,j,j,j],[j,j,j,k,k,k],[i,i,j,j,k,k],[j,j,j,i,i,i],
                           [k,k,k,k,k,k],[k,k,k,i,i,i],[i,i,j,j,k,k],[k,k,k,j,j,j]]),4)
    iter = 1;
    if isquad(T):
        iter=None
    return __xmesh(T,scheme,iter)


