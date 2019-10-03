import os
import os.path
from ..fileio.fileparts    import *
from ..fileio.PNGfile      import *
from ..utility.static_vars import *


@static_vars(cdata=dict())
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

    if name not in palette.cdata:
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'asset')
        filename = [fileparts(f)[1] for f in os.listdir(path)
                    if os.path.isfile(os.path.join(path, f)) and f.endswith('.png')]
        if name not in filename:
            raise RuntimeError('Palette \'{}\' not present. Choose one from the following:\n{}'.format(name, '\n'.join(filename)))
        palette.cdata[name] = import_PNG(os.path.join(path, name + '.png'), device=device).squeeze()
    return palette.cdata[name]


def save_palette(name, cdata):
    """
    Saves the given palette to the asset folder

    Parameters
    ----------
    name : str
        name of the palette
    cdata : Tensor
        the (N,3,) color tensor

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        if name is already in use
    """

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'asset')
    filename = [fileparts(f)[1] for f in os.listdir(path)
                if os.path.isfile(os.path.join(path, f)) and f.endswith('.png')]
    if name in filename:
        raise RuntimeError('Palette \'{}\' already present. Choose another name')
    export_PNG(os.path.join(path, name + '.png'), cdata.unsqueeze(0))
