from torch_geometric.data import Data
from ..utility.lexsort    import *
from ..topology.poly2edge import *
from ..topology.polysort  import *
from .vertex_normal       import *


def mesh2data(P, T, N=None, E=None):
    """
    Converts a given mesh into a torch_geometric Data object

    Parameters
    ----------
    P : Tensor
        the input points set tensor
    T : LongTensor
        the topology tensor
    N : Tensor (optional)
        the vertex normals. If None they will be automatically computed (default is None)
    E : LongTensor
        the edge tensor. If None it will be automatically computed (default is None)

    Returns
    -------
    Data
        a torch_geometric Data object
    """

    return Data(pos       =P.clone(),
                face      =lexsort(polysort(T.clone(), winding=True), dim=0)[0],
                norm      =N.clone() if N is not None else vertex_normal(P, T),
                edge_index=lexsort(poly2edge(T)[0], dim=0)[0] if E is None else E)
