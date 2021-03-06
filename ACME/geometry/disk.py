import torch
from ..utility.linspace    import *
from ..utility.FloatTensor import *
from ..utility.repmat      import *
from ..math.constant       import *
from ..math.pol2cart       import *
from .grid2mesh            import *


def DiskPatch(tile=(8, 8), device='cuda:0'):
    """
    Creates a disk patch

    Parameters
    ----------
    tile : (int,int) (optional)
        the number of divisions of the disk (default is (8,8))
    device : str or torch.device (optional)
        the device the tensors will be stored to (default is 'cuda:0')

    Returns
    -------
    Tensor
        the (3,M,N,) tensor representing the disk
    """

    r = torch.reshape(linspace(0, 1, tile[0] + 1, device=device), (1, tile[0] + 1))
    t = torch.reshape(linspace(0, PI2, tile[1] + 1, device=device), (tile[1] + 1, 1))
    r = repmat(r, (tile[1] + 1, 1))
    t = repmat(t, (1, tile[0] + 1))
    return torch.cat((t.unsqueeze(0), r.unsqueeze(0), torch.zeros_like(t).unsqueeze(0)), dim=0)


def Disk(tile=(8, 8), device='cuda:0'):
    """
    Creates a disk quad mesh

    Parameters
    ----------
    tile : (int,int) (optional)
        the number of divisions of the disk (default is (8,8))
    device : str or torch.device (optional)
        the device the tensors will be stored to (default is 'cuda:0')

    Returns
    -------
    (Tensor,LongTensor,Tensor)
        the point set tensor, the topology tensor, the vertex normals
    """

    T,P = grid2mesh(*tuple(DiskPatch(tile=tile, device=device)))
    P   = pol2cart(P)
    N   = repmat(FloatTensor([[0, 0, 1]], device=device), (row(P), 1))
    return P, T, N
