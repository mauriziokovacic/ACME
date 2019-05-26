from .rot_z import *

def rot2(theta,affine=False,device='cuda:0'):
    """
    Creates a 2D rotation matrix.

    Parameters
    ----------
    theta : float
        the angle of the rotation
    affine : bool (optional)
        if True, the function returns a (3,3) matrix, (2,2) otherwise (default is False)
    device : str or torch.device (optional)
        the device the tensor will be stored to (defaul is 'cuda:0')

    Returns
    -------
    Tensor
        a (2,2) or (3,3) rotation matrix
    """

    R = rot_z(theta,affine=False,device=device)
    if affine:
        return R
    return R[0:2,0:2]
