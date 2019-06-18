from torch_geometric.data    import data
from ACME.topology.poly2poly import *
from .vertex_normal          import *

def mesh2data(P,T,N=None):
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

    Returns
    -------
    Data
        a torch_geometric Data object
    """

    return Data(pos       = P.clone(),
                face      = T.clone(),
                norm      = N.clone() if N is not None else vertex_normal(P,T),
                edge_index= poly2edge(T)[0])
