import torch
from ..math.cross   import *
from ..math.normvec import *


class CameraExtrinsic(object):
    """
    A class representing the camera extrinsic properties

    Attributes
    ----------
    position : Tensor
        the camera position
    target : Tensor
        the camera target
    up_vector : Tensor
        the camera up vector
    device : str or torch.device
        the device to store the tensors to

    Methods
    -------
    look_at(target)
        sets the camera target
    look_from(position)
        sets the camera position
    direction()
        returns the camera direction
    view_matrix()
        returns the current view matrix
    to(**kwargs)
        changes extrinsic dtype and/or device
    """

    def __init__(self, position=(0, 0, 0), target=(0, 0, 1), up_vector=(0, 1, 0), device='cuda:0'):
        """
        Parameters
        ----------
        position : list or tuple (optional)
            the camera position (default is (0, 0, 0))
        target : list or tuple (optional)
            the camera target (default is (0, 0, 1))
        up_vector : list or tuple (optional)
            the camera up vector (default is (0, 1, 0))
        device : str or torch.device (optional)
            the device to store the tensors to (default is 'cuda:0')
        """

        self.position  = torch.tensor(position,  dtype=torch.float, device=device)
        self.target    = torch.tensor(target,    dtype=torch.float, device=device)
        self.up_vector = torch.tensor(up_vector, dtype=torch.float, device=device)
        self.device    = device

    def look_at(self, target):
        """
        Sets the camera target

        Parameters
        ----------
        target : Tensor
            the (3,) target tensor

        Returns
        -------
        CameraExtrinsic
            the extrinsic itself
        """

        self.target = target
        return self

    def look_from(self, position):
        """
        Sets the camera position

        Parameters
        ----------
        position : Tensor
            the (3,) position tensor

        Returns
        -------
        CameraExtrinsic
            the extrinsic itself
        """

        self.position = position
        return self

    def direction(self):
        """
        Returns the camera direction

        Returns
        -------
        Tensor
            the (3,) direction tensor
        """

        return self.target - self.position

    def view_matrix(self):
        """
        Returns the current view matrix

        Returns
        -------
        Tensor
            a (4,4,) view matrix
        """

        z = normr(self.direction().unsqueeze(0))
        x = normr(cross(self.up_vector.unsqueeze(0), z))
        y = cross(z, x)
        p = self.position.unsqueeze(0)
        M = torch.cat((torch.cat((x.t(), y.t(), z.t(), -p.t()), dim=1),
                       torch.tensor([[0, 0, 0, 1]], dtype=torch.float, device=self.device)),
                      dim=0)
        return M

    def to(self, **kwargs):
        """
        Changes the extrinsic dtype and/or device

        Parameters
        ----------
        kwargs : ...

        Returns
        -------
        CameraExtrinsic
            the extrinsic itself
        """

        if 'device' in kwargs:
            self.device = kwargs['device']
        self.position  = self.position.to(**kwargs)
        self.target    = self.target.to(**kwargs)
        self.up_vector = self.up_vector.to(**kwargs)
        return self

    @property
    def device(self):
        return self.device

    @device.setter
    def device(self, value):
        self.device    = value
        self.position  = self.position.to(self.device)
        self.target    = self.target.to(self.device)
        self.up_vector = self.up_vector.to(self.device)
