from torch_geometric.data import data
from topology.poly2poly   import *
from .vertex_normal       import *

def to_data(P,T,N=None):
    """
    Converts a given mesh into a torch_geometric Data object


    """

    data      = Data(
                    pos=P.clone(),
                    face=T.clone(),
                    norm=N.clone() if N is not None else vertex_normal(P,T),
                    edge_index=poly2edge(T))
    data.face = T.clone()
    data.norm = N.clone() if N is not None else vertex_normal(P,T)
    return data
