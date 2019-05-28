import torch
from utility.row          import *
from utility.nop          import *
from math.constant        import *
from math.cos             import *
from math.sin             import *
from math.normvec         import *
from math.cart2sph        import *
from math.sph2rotm        import *
from geometry.octahedron  import *
from geometry.icosahedron import *
from geometry.shape_scale import *
from .mesh2img            import *



def camera_from_polyhedron(polyhedronFcn,camera_distance=1,to_spherical=False, device='cuda:0'):
    """
    Returns the position of a camera lying on the vertices of a given polyhedron

    Parameters
    ----------
    polyhedronFcn : callable
        the polyhedron creation function
    camera_distance : float (optional)
        the camera distance from the origin
    to_spherical : bool (optional)
        if True, converts the coordinates into spherical (default is False)
    device : str or torch.device
        the device the tensors will be stored to (default is 'cuda:0')

    Returns
    -------
    (Tensor,LongTensor)
        the positions and the topology of the camera views
    """

    P,T   = hedronFcn(device=device)[0:2]
    theta = PI/100
    R     = torch.tensor([[1,0,0],[0,cos(theta),-sin(theta)],[0,sin(theta),cos(theta)]],dtype=torch.float,device=device)
    P     = torch.mul(torch.mm(normr(P),torch.t(R)),camera_distance)
    if to_spherical:
        P   = cart2sph(P)
    return P,T



def camera_6(camera_distance=1,to_spherical=False,device='cuda:0'):
    """
    Returns the position of a camera lying on the vertices of an octahedron

    Parameters
    ----------
    camera_distance : float (optional)
        the camera distance from the origin
    to_spherical : bool (optional)
        if True, converts the coordinates into spherical (default is False)
    device : str or torch.device
        the device the tensors will be stored to (default is 'cuda:0')

    Returns
    -------
    (Tensor,LongTensor)
        the positions and the topology of the camera views
    """

    return camera_from_polyhedron(Octahedron,camera_distance=camera_distance,spherical=spherical, device=device)



def camera_12(camera_distance=1,to_spherical=False,device='cuda:0'):
    """
    Returns the position of a camera lying on the vertices of an icosahedron

    Parameters
    ----------
    camera_distance : float (optional)
        the camera distance from the origin
    to_spherical : bool (optional)
        if True, converts the coordinates into spherical (default is False)
    device : str or torch.device
        the device the tensors will be stored to (default is 'cuda:0')

    Returns
    -------
    (Tensor,LongTensor)
        the positions and the topology of the camera views
    """

    return camera_from_polyhedron(Icosahedron,camera_distance=camera_distance,spherical=spherical, device=device)



def camera_18(camera_distance=1,to_spherical=False,device='cuda:0'):
    """
    Returns the position of a camera lying on the vertices of a subdivided octahedron

    Parameters
    ----------
    camera_distance : float (optional)
        the camera distance from the origin
    to_spherical : bool (optional)
        if True, converts the coordinates into spherical (default is False)
    device : str or torch.device
        the device the tensors will be stored to (default is 'cuda:0')

    Returns
    -------
    (Tensor,LongTensor)
        the positions and the topology of the camera views
    """

    return camera_from_polyhedron(Octahedron_2,camera_distance=camera_distance,spherical=spherical, device=device)



def camera_42(camera_distance=1,to_spherical=False,device='cuda:0'):
    """
    Returns the position of a camera lying on the vertices of a subdivided icosahedron

    Parameters
    ----------
    camera_distance : float (optional)
        the camera distance from the origin
    to_spherical : bool (optional)
        if True, converts the coordinates into spherical (default is False)
    device : str or torch.device
        the device the tensors will be stored to (default is 'cuda:0')

    Returns
    -------
    (Tensor,LongTensor)
        the positions and the topology of the camera views
    """

    return camera_from_polyhedron(Icosahedron_2,camera_distance=camera_distance,spherical=spherical, device=device)



def camera_66(camera_distance=1,to_spherical=False,device='cuda:0'):
    """
    Returns the position of a camera lying on the vertices of a twice subdivided octahedron

    Parameters
    ----------
    camera_distance : float (optional)
        the camera distance from the origin
    to_spherical : bool (optional)
        if True, converts the coordinates into spherical (default is False)
    device : str or torch.device
        the device the tensors will be stored to (default is 'cuda:0')

    Returns
    -------
    (Tensor,LongTensor)
        the positions and the topology of the camera views
    """

    return camera_from_polyhedron(Octahedron_3,camera_distance=camera_distance,spherical=spherical, device=device)




def mesh2mvs(renderer,T,P,C=None,Cam=None,out_channel=5,postFcn=nop,pivoting=False):
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
    out_channel : int (optional)
        the number of output channels of the renderer (default is 5)
    postFcn : callable (optional)
        a function to be applied to the Neural Renderer output (defalut is nop)

    Returns
    -------
    Tensor
        the Neural Renderer image in RGBDA format if postFcn is nop
    """

    if Cam is None:
        Cam = camera_18(camera_distance=shape_scale(P)*1.4, spherical=pivoting, device=renderer.device)[0]
    MVS = torch.zeros((row(Cam), out_channel, *(renderer.image_size,)*2), dtype=torch.float, device=renderer.device)
    if pivoting:
        for i in range(0,row(Cam)):
            renderer.eye             = Cam[i,:]
            renderer.light_direction = -Cam[i,:]
            I = mesh2img(renderer,T,P,C=C,postFcn=postFcn)
            MVS[i,:,:,:] = I
    else:
        for i in range(0,row(Cam)):
            M = sph2rotm(Cam[i,:])
            I = mesh2img(renderer,T,torch.mm(P,M),C=C,postFcn=postFcn)
            MVS[i,:,:,:] = I
    return MVS
