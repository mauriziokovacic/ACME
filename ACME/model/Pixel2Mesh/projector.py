import torch
from ...render.camera import *


class Projector(torch.nn.Module):
    """
    A class representing the projection operation

    Attributes
    ----------
    cam : Camera
        the camera used to render the input data

    Methods
    -------
    forward(x)
        computes the projection of the input points
    """

    def __init__(self, cam=Camera()):
        """
        Parameters
        ----------
        cam : Camera
            the camera used for the projection
        """

        super(Projector, self).__init__()
        self.camera = cam

    def forward(self, x):
        """
        Computes the projection of the input points

        Parameters
        ----------
        x : Data
            the input model to project

        Returns
        -------
        Tensor
            the (N,2,) texture coordinates tensor
        """

        return self.camera.project(x, pixels=False)[:, :-1]
