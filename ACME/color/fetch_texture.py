import torch
from ACME.utility.col       import *
from ACME.utility.numel     import *
from ACME.utility.to_column import *



def fetch_texture1D(texture,t,mode='bilinear'):
    """
    Fetches an input texture using the given UVs in range [0,1]

    Parameters
    ----------
    texture : Tensor
        the input texture tensor with shape (N,3,)
    t : Tensor
        the input parameter tensor with shape (N,)
    mode : str (optional)
        interpolation method. Only 'bilinear' or 'nearest' are accepted

    Returns
    -------
    Tensor
        the fetched data from the input texture
    """

    return torch.t(torch.reshape(torch.nn.functional.grid_sample(
                torch.reshape(torch.t(texture),(1,3,-1,1)),
                torch.reshape(torch.cat((torch.zeros(numel(t),1,
                                                     dtype=torch.float,
                                                     device=texture.device),
                                         to_column(t*2-1)),dim=1),(1,1,-1,2)),
                mode=mode,
                padding_mode='border'),(3,numel(t))))



def fetch_texture2D(texture,uv,mode='bilinear'):
    """
    Fetches an input texture using the given UVs in range [0,1]

    Parameters
    ----------
    texture : Tensor
        the input texture tensor with shape (C,W,H,)
    uv : Tensor
        the input UV tensor with shape (N,2,)
    mode : str (optional)
        interpolation method. Only 'bilinear' or 'nearest' are accepted

    Returns
    -------
    Tensor
        the fetched data from the input texture
    """

    return torch.reshape(torch.nn.functional.grid_sample(texture.unsqueeze(0),
                                                         torch.reshape(uv*2-1,(1,1,-1,2)),
                                                         mode=mode,
                                                         padding_mode='border').squeeze(0),(-1,texture.shape[0]))



def fetch_texture3D(texture,uv,mode='bilinear'):
    """
    Fetches an input texture using the given UVs in range [0,1]

    Parameters
    ----------
    texture : Tensor
        the input texture tensor with shape (C,W,H,D)
    uv : Tensor
        the input UV tensor with shape (N,3,)
    mode : str (optional)
        interpolation method. Only 'bilinear' or 'nearest' are accepted

    Returns
    -------
    Tensor
        the fetched data from the input texture
    """

    return torch.reshape(torch.nn.functional.grid_sample(texture.unsqueeze(0),
                                                         torch.reshape(uv*2-1,(1,1,1,-1,3)),
                                                         mode=mode,
                                                         padding_mode='border').squeeze(0),(-1,texture.shape[0]))
