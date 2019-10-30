from ..utility.cat      import *
from ..utility.issparse import *
from ..math.unitvec     import *


def add_constraints(A, hard=None, soft=None):
    """
    Adds hard and soft constraints to the given constraints matrix

    Parameters
    ----------
    A : Tensor or SparseTensor
        the (N,M,) constraints matrix
    hard : iterable
        the hard constraints indices
    soft : LongTensor
        the (S,) indices tensor for the soft constraints

    Returns
    -------
    Tensor
        the (N+S,M,) constraints matrix
    """

    M = A.clone()
    if hard is not None:
        S = unitvec(A.size(-1), hard, sparse=False, dtype=torch.float, device=A.device)
        if issparse(A):
            M = M.to_dense()
        M[hard] = S
    if soft is not None:
        S = unitvec(A.size(-1), soft, sparse=True, dtype=torch.float, device=A.device)
        M = cat((M, S), dim=-2)
    return M
