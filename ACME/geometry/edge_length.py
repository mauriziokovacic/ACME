from ACME.math.norm          import *
from ACME.topology.poly2edge import *

def edge_length(P,T,dim=1):
    """
    Returns the edge lenghts of the given input topology

    Parameters
    ----------
    P : Tensor
        the input point set tensor
    T : LongTensor
        the topology tensor
    dim : int (optional)
        the dimnsion along the length is computed (default is 1)

    Returns
    -------
    Tensor
        the edge lenghts of the input topology
    """

    E     = poly2edge(T)[0]
    Pi,Pj = P[E]
    return distance(Pi,Pj,dim=dim)
