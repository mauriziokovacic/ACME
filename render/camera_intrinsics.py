import torch
from ..math.deg2rad import *
from ..math.tan     import *


class CameraIntrinsic(object):
    """
    A class representing the camera intrinsic

    Attributes
    ----------
    fov : float
        the camera field of view angle in degrees
    near : float
        the near clipping plane distance
    far : float
        the far clipping plane distance
    image_size : tuple or list
        the image width and height
    projection : str
        the camera projection type ('orthographic' or 'perspective')
    device : str or torch.device
        the device to store the tensors to

    Methods
    -------
    aspect()
        returns the aspect ratio of the image
    projection_matrix()
        returns the current projection matrix
    orthographic_matrix()
        returns the current orthographic matrix
    perspective_matrix()
        returns the current perspective matrix
    to(**kwargs)
        changes the intrinsic device
    """

    def __init__(self, fov=30, near=0.1, far=10, image_size=(256, 256), projection='perspective', device='cuda:0'):
        """
        Parameters
        ----------
        fov : float (optional)
            the camera field of view in degrees (default is 30)
        near : float (optional)
            the camera near clipping plane distance (default is 0.1)
        far : float (optional)
            the camera far clipping plane distance (default is 10)
        image_size : list or tuple (optional)
            the image width and height (default is (256, 256))
        projection : str (optional)
            the projection type (default is 'perspective')
        device : str or torch.device (optional)
            the device to store the tensors to (default is 'cuda:0')
        """

        self.fov        = fov
        self.near       = near
        self.far        = far
        self.image_size = image_size
        self.projection = projection
        self.device     = device

    def aspect(self):
        """
        Returns the aspect ratio of the image

        Returns
        -------
        float
            the aspect ratio
        """
        return self.image_size[0] / self.image_size[1]

    def projection_matrix(self):
        """
        Returns the current projection matrix

        Returns
        -------
        Tensor
            a (4,4,) projection matrix
        """

        if self.projection == 'orthographic':
            return self.orthographic_matrix()
        if self.projection == 'perspective':
            return self.perspective_matrix()
        raise ValueError('Unknown projection type.')

    def orthographic_matrix(self):
        """
        Returns the orthographic projection matrix

        Returns
        -------
        Tensor
            a (4,4,) projection matrix
        """

        fov = deg2rad(self.fov)
        M = torch.zeros(4, 4, device=self.device)
        M[0, 0] = 1 / (self.aspect() * tan(fov / 2))
        M[1, 1] = 1 / tan(fov / 2)
        M[2, 2] = 2 / (self.far - self.near)
        M[2, 3] = -(self.far + self.near) / (self.far - self.near)
        M[3, 3] = 1
        return M

    def perspective_matrix(self):
        """
        Returns the perspective projection matrix

        Returns
        -------
        Tensor
            a (4,4,) projection matrix
        """

        fov = deg2rad(self.fov)
        M = torch.zeros(4, 4, device=self.device)
        M[0, 0] = 1 / (self.aspect() * tan(fov / 2))
        M[1, 1] = 1 / tan(fov / 2)
        M[2, 2] = (self.far + self.near) / (self.far - self.near)
        M[2, 3] = -2 * (self.far * self.near) / (self.far - self.near)
        M[3, 2] = 1
        return M

    def to(self, **kwargs):
        """
        Changes the the intrinsic device

        Parameters
        ----------
        kwargs : ...

        Returns
        -------
        CameraIntrinsic
            the intrinsic itself
        """

        if 'device' in kwargs:
            self.device = kwargs['device']
        return self
