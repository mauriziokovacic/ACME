import numpy as np
import torch

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
        i  = np.lexsort(c,axis=0)
        d  = np.reshape(np.append(np.ones((1,col(C)))+tol,np.abs(np.diff(C[i],axis=0))),(-1,col(A)))
        tf = np.sum(d>tol,1)>0
        n  = row(A)
    else:
        C  = np.ravel(C)
        i  = np.argsort(C)
        d  = np.append(1+tol,np.diff(C[i]))
        tf = d>(tol*np.max(np.abs(C)))
        n  = np.size(A)
    ia = i[tf]
    C  = C[ia]
    ic = np.zeros(n,dtype=int)
    j = -1
    for x in range(0,n):
        if tf[x]:
            j += 1
        ic[i[x]] = j
    return torch.from_numpy(C ).to(dtype=A.dtype,   device=A.device),\
           torch.from_numpy(ia).to(dtype=torch.long,device=A.device),\
           torch.from_numpy(ic).to(dtype=torch.long,device=A.device)
