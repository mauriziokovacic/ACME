from ..utility.clamp    import *
from ..utility.isstring import *
from ..math.normalize   import *
from .color_gradient    import *
from .fetch_texture     import *
from .palette           import *


class ColorMap(object):
    """
    A class representing a color map

    Attributes
    ----------
    cdata : Tensor
        the color data
    name : str
        the name of the color map
    device : str or torch.device
        the device to store the tensors to

    Methods
    -------
    fetch(tensor, cres, casix)
        returns the colors for the given input data
    to(device)
        moves the color map to the given device
    """

    def __init__(self, cdata='parula', cres=64, caxis=None, name='ColorMap', device='cuda:0'):
        """
        Parameters
        ----------
        cdata : Tensor or str (optional)
            a (N,3,) color tensor or the name of a color palette (default is 'parula')
        cres : int (optional)
            the resolution of the color map
        caxis : Range (optional)
            the accepted [min, max] range for the tensor data.
            If None, [min(tensor), max(tensor)] will be used (default is None)
        name : str (optional)
            the name of the color map (default is 'ColorMap')
        device : str or torch.device (optional)
            the device the tensors are stored to (default is 'cuda:0')
        """

        if isstring(cdata):
            self.cdata = palette(cdata, device=device)
        else:
            self.cdata = cdata
        self.name   = name
        self.device = device

    def fetch(self, tensor):
        """
        Returns the colors for the given input data

        Parameters
        ----------
        tensor : Tensor
            a (N,) tensor

        Returns
        -------
        Tensor
            the (N,3,) color tensor
        """

        cdata = self.cdata
        if self.cres != self.cdata.size(0):
            cdata = color_gradient(cdata, self.cres)
        data = tensor.clone().to(self.device)
        if self.caxis is not None:
            data = clamp(data, inf=self.caxis[0], sup=self.caxis[1])
        return fetch_texture1D(cdata, normalize(data))

    def to(self, device):
        """
        Moves the color map to the given device

        Parameters
        ----------
        device : str or torch.device
            the device to store the tensors to

        Returns
        -------
        ColorMap
            the color map itself
        """

        self.device = device
        return self

    def __setattr__(self, key, value):
        self.__dict__[key] = value
        if key == 'device':
            self.cdata.to(device=self.device)

    def __call__(self, *args, **kwargs):
        return self.fetch(*args, **kwargs)

    def __repr__(self):
        return self.name