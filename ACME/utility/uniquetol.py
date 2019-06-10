import numpy
import torch
from .numpy2torch import *

def uniquetol(A,tol=10e-4,ByRows=False):
    """
    Returns the unique elements in A using tolerance tol.

    This function returns a triplet in the form of (C,iA,iC), such that:
        - C = A[iA]
        - A = C[iC]

    Parameters
    ----------
    A : Tensor
        the input tensor
    tol : int or float (optional)
        the tolerance used in the process (default is 10e-4)
    ByRows : bool (optional)
        if True treats the rows as a single entity

    Returns
    -------
    (Tensor,LongTensor,LongTensor)
        returns the tensor of unique values/rows, their indices within the input tensor and the input tensor indices within the unique
    """

    C  = A.cpu().numpy()
    if ByRows:
        c  = []
        for i in range(col(A)-1,-1,-1):
            c = c+[C[:,i]]
        i  = numpy.lexsort(c,axis=0)
        d  = numpy.reshape(numpy.append(numpy.ones((1,col(C)))+tol,numpy.abs(numpy.diff(C[i],axis=0))),(-1,col(A)))
        tf = numpy.sum(d>tol,1)>0
        n  = row(A)
    else:
        C  = numpy.ravel(C)
        i  = numpy.argsort(C)
        d  = numpy.append(1+tol,numpy.diff(C[i]))
        tf = d>(tol*numpy.max(numpy.abs(C)))
        n  = numpy.size(A)
    ia = i[tf]
    C  = C[ia]
    ic = numpy.zeros(n,dtype=int)
    j = -1
    for x in range(0,n):
        if tf[x]:
            j += 1
        ic[i[x]] = j
    return numpy2torch(C, dtype=A.dtype,   device=A.device),\
           numpy2torch(ia,dtype=torch.long,device=A.device),\
           numpy2torch(ic,dtype=torch.long,device=A.device)
