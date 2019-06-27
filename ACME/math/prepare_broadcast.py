import torch

def prepare_broadcast(A, B):
    """
    Prepares the tensors for a broadcasting operation of A onto B

    Parameters
    ----------
    A : Tensor
        a (N,F,) tensor
    B : Tensor
        a (M,F,) tensor

    Returns
    -------
    (Tensor, Tensor)
        the first (N,1,F,) tensor and the second (N,M,F,) tensor
    """

    return A.view(A.size(0), 1, A.size(1)),\
           B.view(-1, *B.size()).expand(A.size(0), -1, B.size(1))
