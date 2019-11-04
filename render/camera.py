import torch
from ..math.cart2homo   import *
from ..math.homo2cart   import *
from .camera_extrinsics import *
from .camera_intrinsics import *


class Camera(object):
    """
    A class representing a camera

    Attributes
    ----------
    extrinsic : CameraExtrinsic
        the camera extrinsic
    intrinsic : CameraIntrinsic
        the camera intrinsic
    name : str
        the camera name
    device : str or torch.device
        the device to store the tensors to

    Methods
    -------
    project(P)
        projects the given 3D points into the 2D image
    unproject(Q)
        unprojects the given 2D points + depth to 3D space
    to(**kwargs)
        changes the camera dtype and/or device
    """

    def __init__(self,
                 extrinsic=CameraExtrinsic(),
                 intrinsic=CameraIntrinsic(),
                 name='Camera', device='cuda:0'):
        """
        Parameters
        ----------
        extrinsic : CameraExtrinsic (optional)
            the camera extrinsic (default is CameraExtrinsic())
        intrinsic : CameraIntrinsic (optional)
            the camera intrinsic (default is CameraIntrinsic())
        name : str (optional)
            the name of the camera (default is 'Camera')
        device : str or torch.device (optional)
            the device to store the tensors to
        """

        self.extrinsic = extrinsic
        self.intrinsic = intrinsic
        self.name      = name
        self.device    = device

    def project(self, P, pixels=True, dim=-1):
        """
        Projects the given 3D points into the 2D image

        Parameters
        ----------
        P : Tensor
            a (N,3,) points set tensor
        pixels : bool (optional)
            if True the UVs are returned in floating point pixel coordinates

        Returns
        -------
        Tensor
            a (N,3,) tensor containing UVs and depth
        """

        s = 0.5
        if pixels:
            # Image width and height
            w = self.intrinsic.image_size[0] - 1
            h = self.intrinsic.image_size[1] - 1
            t = torch.ones(P.ndimension(), dtype=torch.long, device=P.device)
            t[dim] = -1
            # Normalization factor (bring the coordinates from [-1,1] to [0, w], [0, h] and [0, 1] respectively)
            s *= torch.tensor([w, h, 1], dtype=torch.float, device=self.device).view(*t)
        # Transform the points into homogeneous coordinates, transform them into camera space and then project them
        UVd = torch.matmul(cart2homo(P, w=1, dim=dim),
                           torch.transpose(torch.matmul(self.intrinsic.projection_matrix(),
                                                        self.extrinsic.view_matrix()),
                                           -1, -2)
                           )
        # Bring the points into normalized homogeneous coordinates and normalize their values
        return homo2cart(UVd, dim=dim) * s + s

    def unproject(self, UVd, pixels=True, dim=-1):
        """
        Unprojects the given 2D points + depth to 3D space

        Parameters
        ----------
        UVd : Tensor
            a (N,3,) points set tensor consisting of UVs and depth values
        pixels : bool (optional)
            if True, the UVs are treated as floating point pixel coordinates

        Returns
        -------
        Tensor
            a (N,3,) points set tensor
        """

        s = 2
        if pixels:
            # Image width and height
            w = self.intrinsic.image_size[0] - 1
            h = self.intrinsic.image_size[1] - 1
            t = torch.ones(UVd.ndimension(), dtype=torch.long, device=UVd.device)
            t[dim] = -1
            # Normalization factor (brings the coordinates to [-1, 1])
            s /= torch.tensor([w, h, 1], dtype=torch.float, device=self.device).view(*t)
        # Change the points domain, transform them into homogeneous, and invert the projection process
        P = torch.matmul(cart2homo(UVd * s - 1, w=1, dim=dim),
                         torch.inverse(torch.matmul(self.intrinsic.projection_matrix(),
                                                    self.extrinsic.view_matrix())))
        # Normalize the coordinates
        return homo2cart(P, dim=dim)

    def to(self, **kwargs):
        """
        Changes the camera dtype and/or device

        Parameters
        ----------
        kwargs : ...

        Returns
        -------
        Camera
            the camera itself
        """

        if 'device' in kwargs:
            self.device = kwargs['device']
        self.extrinsic.to(**kwargs)
        self.intrinsic.to(**kwargs)
        return self

    def __setattr__(self, key, value):
        self.__dict__[key] = value
        if key == 'device':
            self.extrinsic.to(device=self.device)
            self.intrinsic.to(device=self.device)


