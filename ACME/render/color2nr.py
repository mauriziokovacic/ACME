import torch
from ..utility.row       import *
from ..utility.col       import *
from ..utility.repmat    import *
from ..color.color2float import *


def color2nr(T, C, texture_size=2, dtype=torch.float32):
    """
    Converts an RGB color tensor into the Neural Renderer texture

    Parameters
    ----------
    T : LongTensor
        the triangle topology tensor
    C : Tensor or Uint8Tensor
        the RGB color tensor
    texture_size : int (optional)
        the Neural Renderer texture size (default is 2)
    dtype : type (optional)
        the Neural Renderer texture type (default is torch.float32)

    Returns
    -------
    Tensor
        the Neural Renderer texture
    """

    nr = torch.zeros(1, col(T), *(texture_size,)*3, 3, dtype=dtype, device=T.device)
    c  = color2float(C)
    if row(C) == 1:
        c = repmat(c, (col(T), 1))
    if col(T) == row(C):
        nr = nr.permute(0, 2, 3, 4, 1, 5)
        nr[0, :, :, :] = c
        nr = nr.permute(0, 4, 1, 2, 3, 5)
    else:
        x = texture_size-1
        Ci, Cj, Ck = c[T]
        nr[0, :, 0, 0, 0, :] = (Ci+Cj+Ck)/3
        nr[0, :, x, 0, 0, :] = Ci
        nr[0, :, 0, x, 0, :] = Cj
        nr[0, :, x, x, 0, :] = (Ci+Cj)*0.5
        nr[0, :, 0, 0, x, :] = Ck
        nr[0, :, x, 0, x, :] = (Ck+Ci)*0.5
        nr[0, :, 0, x, x, :] = (Cj+Ck)*0.5
    return nr
