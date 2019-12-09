import numpy
import torch
from PIL                   import Image
from ..utility.depth       import *
from ..utility.isnumpy     import *
from ..utility.istorch     import *
from ..utility.numpy2torch import *
from ..utility.torch2numpy import *


def export_PNG(filename, I):
    """
    Exports a given tensor in range [0,1] to the specified PNG file

    Parameters
    ----------
    filename : str
        the file name comprehensive of extesion
    I : Tensor
        the (W,H,C,) tensor to export
    """

    type = 'L'
    if depth(I) == 3:
        type = 'RGB'
    if depth(I) == 4:
        type = 'RGBA'
    if istorch(I):
        Image.fromarray(torch2numpy(I * 255).astype(numpy.uint8), type).save(filename)
    if isnumpy(I):
        Image.fromarray((I * 255).astype(numpy.uint8), type).save(filename)
    return


def import_PNG(filename, device='cuda:0'):
    """
    Imports the given PNG file into a tensor

    Parameters
    ----------
    filename : str
        the file to import
    device : str or torch.device (optional)
        the device to store the tensor to (default is 'cuda:0')

    Returns
    -------
    Tensor
        a (W,H,C,) tensor in range [0,1]
    """

    return numpy2torch(numpy.array(Image.open(filename), dtype=float) / 255, dtype=torch.float, device=device)
