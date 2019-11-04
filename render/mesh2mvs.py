import torch
from ..utility.identity     import *
from ..geometry.shape_scale import *
from ..math.sph2rotm        import *
from .mesh2img              import *
from .cameras               import *


def mesh2mvs(renderer, T, P, C=None, Cam=None, postFcn=identity, pivoting=False):
    """
    Renders a multi view stack of images of the input mesh with the given renderer

    Parameters
    ----------
    renderer : Neural Renderer
        an instance of the Neural Renderer
    T : LongTensor
        the topology tensor
    P : Tensor
        the points set tensor
    C : Tensor or Uint8Tensor (optional)
        the RGB color tensor. If None the mesh will be colored white (default is None)
    Cam : Tensor (optional)
        the view points to render the mesh from. If None, 18 distribuited points will be choosen (default is None)
    postFcn : callable (optional)
        a function to be applied to the Neural Renderer output (defalut is identity)
    pivoting : bool (optional)
        if True rotates the mesh instead of moving the camera, False otherwise (default is False)

    Returns
    -------
    Tensor
        the Neural Renderer image in RGBDA format if postFcn is identity
    """

    if Cam is None:
        Cam = camera_18(camera_distance=shape_scale(P)*1.4, to_spherical=pivoting, device=renderer.device)[0]
    if not pivoting:
        def viewFcn(c):
            renderer.eye             = c
            renderer.light_direction = -c
            return mesh2img(renderer, T, P, C=C, postFcn=postFcn)
    else:
        def viewFcn(c):
            return mesh2img(renderer, T, torch.mm(P, sph2rotm(c)), C=C, postFcn=postFcn)
    return torch.cat(tuple(viewFcn(c) for c in Cam), dim=0)
