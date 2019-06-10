import torch
from ACME.utility.row          import *
from ACME.utility.nop          import *
from ACME.math.constant        import *
from ACME.math.cos             import *
from ACME.math.sin             import *
from ACME.math.normvec         import *
from ACME.math.cart2sph        import *
from ACME.math.sph2rotm        import *
from ACME.geometry.octahedron  import *
from ACME.geometry.icosahedron import *
from ACME.geometry.shape_scale import *
from .mesh2img                 import *



def camera_from_polyhedron(polyhedronFcn,camera_distance=1,to_spherical=False, device='cuda:0'):
    """
    Returns the positions of a camera lying on the vertices of a given polyhedron

    Parameters
    ----------
    polyhedronFcn : callable
        the polyhedron creation function
    camera_distance : float (optional)
        the camera distance from the origin (default is 1)
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
    Returns the positions of a camera lying on the vertices of an octahedron

    Parameters
    ----------
    camera_distance : float (optional)
        the camera distance from the origin (default is 1)
    to_spherical : bool (optional)
        if True, converts the coordinates into spherical (default is False)
    device : str or torch.device
        the device the tensors will be stored to (default is 'cuda:0')

    Returns
    -------
    (Tensor,LongTensor)
        the positions and the topology of the camera views
    """

    return camera_from_polyhedron(Octahedron,camera_distance=camera_distance,to_spherical=to_spherical, device=device)



def camera_12(camera_distance=1,to_spherical=False,device='cuda:0'):
    """
    Returns the positions of a camera lying on the vertices of an icosahedron

    Parameters
    ----------
    camera_distance : float (optional)
        the camera distance from the origin (default is 1)
    to_spherical : bool (optional)
        if True, converts the coordinates into spherical (default is False)
    device : str or torch.device
        the device the tensors will be stored to (default is 'cuda:0')

    Returns
    -------
    (Tensor,LongTensor)
        the positions and the topology of the camera views
    """

    return camera_from_polyhedron(Icosahedron,camera_distance=camera_distance,to_spherical=to_spherical, device=device)



def camera_18(camera_distance=1,to_spherical=False,device='cuda:0'):
    """
    Returns the positions of a camera lying on the vertices of a subdivided octahedron

    Parameters
    ----------
    camera_distance : float (optional)
        the camera distance from the origin (default is 1)
    to_spherical : bool (optional)
        if True, converts the coordinates into spherical (default is False)
    device : str or torch.device
        the device the tensors will be stored to (default is 'cuda:0')

    Returns
    -------
    (Tensor,LongTensor)
        the positions and the topology of the camera views
    """

    return camera_from_polyhedron(Octahedron_2,camera_distance=camera_distance,to_spherical=to_spherical, device=device)



def camera_42(camera_distance=1,to_spherical=False,device='cuda:0'):
    """
    Returns the positions of a camera lying on the vertices of a subdivided icosahedron

    Parameters
    ----------
    camera_distance : float (optional)
        the camera distance from the origin (default is 1)
    to_spherical : bool (optional)
        if True, converts the coordinates into spherical (default is False)
    device : str or torch.device
        the device the tensors will be stored to (default is 'cuda:0')

    Returns
    -------
    (Tensor,LongTensor)
        the positions and the topology of the camera views
    """

    return camera_from_polyhedron(Icosahedron_2,camera_distance=camera_distance,to_spherical=to_spherical, device=device)



def camera_66(camera_distance=1,to_spherical=False,device='cuda:0'):
    """
    Returns the positions of a camera lying on the vertices of a twice subdivided octahedron

    Parameters
    ----------
    camera_distance : float (optional)
        the camera distance from the origin (default is 1)
    to_spherical : bool (optional)
        if True, converts the coordinates into spherical (default is False)
    device : str or torch.device
        the device the tensors will be stored to (default is 'cuda:0')

    Returns
    -------
    (Tensor,LongTensor)
        the positions and the topology of the camera views
    """

    return camera_from_polyhedron(Octahedron_3,camera_distance=camera_distance,to_spherical=to_spherical, device=device)



def camera_n(n,camera_distance=1,to_spherical=False,device='cuda:0'):
    """
    Returns the positions of a camera lying on the vertices of a equilateral polygon on the XY plane

    Parameters
    ----------
    n : float
        the number of vertices in the polygon
    camera_distance : float (optional)
        the camera distance from the origin (default is 1)
    to_spherical : bool (optional)
        if True, converts the coordinates into spherical (default is False)
    device : str or torch.device
        the device the tensors will be stored to (default is 'cuda:0')

    Returns
    -------
    (Tensor,LongTensor)
        the positions and the topology of the camera views
    """

    P = torch.mul(equilateral_polygon(n,device=device),camera_distance)
    T = torch.t(torch.cat((indices(0,n-2,device=device),indices(1,n-1,device=device)),dim=1))
    if to_spherical:
        P = cart2sph(P)
    return P,T


def mesh2mvs(renderer,T,P,C=None,Cam=None,postFcn=nop,culling=None,pivoting=False):
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
        a function to be applied to the Neural Renderer output (defalut is nop)
    culling : str (optional)
        culling type, being either 'back' or 'front'. If None it won't be applied (default is None)
    pivoting : bool (optional)
        if True rotates the mesh instead of moving the camera, False otherwise (default is False)

    Returns
    -------
    Tensor
        the Neural Renderer image in RGBDA format if postFcn is nop
    """

    if Cam is None:
        Cam = camera_18(camera_distance=shape_scale(P)*1.4, to_spherical=pivoting, device=renderer.device)[0]
    if pivoting:
        def viewFcn(c):
            renderer.eye             = c
            renderer.light_direction = -c
            return mesh2img(renderer,T,P,C=C,postFcn=postFcn,culling=culling)
    else:
        def viewFcn(c):
            return mesh2img(renderer,T,torch.mm(P,sph2rotm(c)),C=C,postFcn=postFcn,culling=culling)
    return torch.cat(tuple(viewFcn(c).unsqueeze(0) for c in Cam),dim=0)
