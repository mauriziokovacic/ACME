import torch

def adjacency(E,W,size=None):
    """
    Computes the adjacency matrix with the given weights for the specified edges

    Parameters
    ----------
    E : LongTensor
        the (2,N,) edge topology tensor
    W : Tensor
        the (N,) edges weights tensor
    size : int (optional)
        the adjacency matrix size. If None it will be automatically computed (default is None)

    Returns
    -------
    Tensor
        the adjacency matrix
    """

    if size is None:
        size = max(I.max()+1,J.max()+1)
    A = torch.zeros(size,size,dtype=W.dtype,device=W.device)
    for i,j,w in zip(*tuple(torch.t(E)),W):
        A[i,j] += w
    return A
