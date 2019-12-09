import torch
from torch.nn.functional import grid_sample
from ..utility.numel     import *
from ..utility.to_column import *


def fetch_texture1D(texture, uv, mode='bilinear'):
    """
    Fetches an input texture using the given UVs in range [0,1]

    Parameters
    ----------
    texture : Tensor
        the input texture tensor with shape (W,) or (W,C,) or (B,W,C,)
    uv : Tensor
        the input parameter tensor with shape (N,) or (B,N,)
    mode : str (optional)
        interpolation method. Only 'bilinear' or 'nearest' are accepted. Only 'bilinear' retains the gradient.

    Returns
    -------
    Tensor
        the (N,C,) or (B,N,C,) fetched data from the input texture
    """

    return torch.t(torch.reshape(grid_sample(
                torch.reshape(torch.t(texture), (1, 3, -1, 1)),
                torch.reshape(torch.cat((torch.zeros(numel(uv), 1,
                                                     dtype=torch.float,
                                                     device=texture.device),
                                         to_column(uv*2-1)), dim=1), (1, 1, -1, 2)),
                mode=mode,
                padding_mode='border'), (3, numel(uv))))


def fetch_texture2D(texture, uv, mode='bilinear'):
    """
    Fetches an input texture using the given UVs in range [0,1]

    Parameters
    ----------
    texture : Tensor
        the input texture tensor with shape (W,H,) or (C,W,H,) or (B,C,W,H,)
    uv : Tensor
        the input UV tensor with shape (N,2,) or (B,N,2,)
    mode : str (optional)
        interpolation method. Only 'bilinear' or 'nearest' are accepted. Only 'bilinear' retains the gradient.

    Returns
    -------
    Tensor
        the (N,C,) or (B,N,C,) fetched data from the input texture
    """

    t = texture
    # Add a dumb channel dimension
    if t.ndimension() < 3:
        t = t.view(1, *t.shape)
    # Add a dumb batch dimension
    if t.ndimension() < 4:
        t = t.view(1, *t.shape)
    # Normalize in [-1, 1]
    c = uv * 2 - 1
    # Add a dumb channel dimension
    if c.ndimension() < 3:
        c = c.expand(t.shape[0], *c.shape)
    # Add a dumb batch dimension
    if c.ndimension() < 4:
        c = c.view(c.shape[0], 1, *c.shape[1:])
    # Make batch dimensions equal
    if t.shape[0] < c.shape[0]:
        t = t.expand(c.shape[0], *t.shape[1:])
    return torch.transpose(grid_sample(t, c, mode=mode, padding_mode='border').squeeze(), -1, -2)


def fetch_texture3D(texture, uv, mode='bilinear'):
    """
    Fetches an input texture using the given UVs in range [0,1]

    Parameters
    ----------
    texture : Tensor
        the input texture tensor with shape (C,W,H,D)
    uv : Tensor
        the input UV tensor with shape (N,3,)
    mode : str (optional)
        interpolation method. Only 'bilinear' or 'nearest' are accepted. Only 'bilinear' retains the gradient.

    Returns
    -------
    Tensor
        the fetched data from the input texture
    """

    return torch.reshape(grid_sample(texture.unsqueeze(0),
                                     torch.reshape(uv*2-1, (1, 1, 1, -1, 3)),
                                     mode=mode,
                                     padding_mode='border').squeeze(0), (-1, texture.shape[0]))
