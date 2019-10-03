import os
import os.path
from ..fileio.fileparts import *
from ..fileio.PNGfile import *


def palette(name, device='cuda:0'):
    """
    Returns the color palette with the specified name

    The color palettes are contained in the color/asset folder

    Parameters
    ----------
    name : str
        the name of the color palette
    device : str or torch.device (optional)
        the device the tensor will be stored to (default is 'cuda:0')

    Returns
    -------
    Tensor
        the (N,3,) color tensor

    Raises
    ------
    RuntimeError
        if the name is unknown
    """

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'asset')
    filename = [fileparts(f)[1] for f in os.listdir(path)
                if os.path.isfile(os.path.join(path, f)) and f.endswith('.png')]
    if name not in filename:
        raise RuntimeError('Palette \'{}\' not present. Choose one from the following:\n{}'.format(name, '\n'.join(filename)))
    return import_PNG(os.path.join(path, name + '.png'), device=device).squeeze()
