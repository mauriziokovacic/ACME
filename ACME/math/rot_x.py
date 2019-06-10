from ACME.utility.FloatTensor import *
from .axang2rotm              import *
from .rotm2affine             import *

def rot_x(theta,affine=False,device='cuda:0'):
    """
    Creates a 3D rotation matrix around x axis with the specified angle in radians

    Parameters
    ----------
    theta : float
        angle in radians
    affine : bool (optional)
        if True the output will be a (4,4) matrix, (3,3) otherwise (default is False)
    device : str or torch.device (optional)
        the device where the tensor will be stored to (default is 'cuda:0')

    Returns
    -------
    Tensor
        a (3,3) or (4,4) rotation matrix
    """

    R = axang2rotm(FloatTensor([1,0,0],device=device),theta)
    if affine:
        R = rotm2affine(R)
    return R
