import torch
from ..utility.row        import *
from ..utility.indices    import *
from ..topology.ind2poly  import *
from ..topology.poly2edge import *



def spherical_helix(t,c=0.1):
    """
    Creates a spherical helix

    Parameters
    ----------
    t : Tensor
        a (N,) tensor with the parametric coordinates of the helix points in [0,1]
    c : float
        the helix constant. The smaller, the more spires

    Returns
    -------
    (Tensor,LongTensor)
        the points set tensor and the edge topology
    """

    theta = t*2-1
    P     = torch.cat((torch.sqrt(1-theta**2) * torch.cos(theta/c),
                       torch.sqrt(1-theta**2) * torch.sin(theta/c),
                       theta),dim=1)
    E     = poly2edge(ind2edge(indices(0,row(P)-2,1,device=t.device),indices(1,row(P)-1,device=t.device)))[0]
    return P,poly2edge(E)[0]
