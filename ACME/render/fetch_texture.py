import torch
from utility.col import *

def fetch_texture(texture,uv,mode='bilinear'):
    """
    Fetches an input texture using the given UVs

    Parameters
    ----------
    texture : Tensor
        the input texture tensor with shape (C,W,H,) or (C,W,H,D)
    uv : Tensor
        the input UV tensor with shape (N,2,) or (N,3,)
    mode : str (optional)
        interpolation method. Only 'bilinear' or 'nearest' are accepted

    Returns
    -------
    Tensor
        the fetched data from the input texture
    """

    return torch.nn.functional.grid_sample(texture.unsqueeze(0),
                                           torch.reshape(uv*2-1,(1,1,-1,col(uv))),
                                           mode=mode,
                                           padding_mode='border').squeeze(0)
