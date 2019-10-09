import torch
from ..math.tan                     import *
from ..math.cart2sph                import *
from ..math.sph2cart                import *
from ..math.sph2rotm                import *
from ..topology.ind2poly            import *
from ..topology.poly2edge           import *
from ..topology.poly2unique         import *
from ..geometry.equilateral_polygon import *
from ..geometry.octahedron          import *
from ..geometry.icosahedron         import *
from ..geometry.sphere              import *
from ..geometry.soup2mesh           import *


def view_matrix(cam, target, up):
    """
    Returns the view matrix from the camera position, target and up vector

    Parameters
    ----------
    cam : Tensor
        the (1,3,) camera position tensor
    target : Tensor
        the (1,3,) target position tensor
    up : Tensor
        the (1,3,) up tensor

    Returns
    -------
    Tensor
        a (4,4,) view matrix
    """

    M        = eye(4, device=cam.device)
    dir      = normr(target-cam)
    vr       = cross(dir, up)
    vup      = cross(dir, vr)
    M[:3, :] = torch.cat((vr.t(), vup.t(), dir.t(), cam.t()), dim=1)
    return M


def perspective_matrix(aspect, fov, near, far, device='cuda:0'):
    """
    Returns the perspective projection matrix from the camera settings

    Parameters
    ----------
    aspect : float
        the aspect ratio of the final image
    fov : float
        the lens angle aperture (in radians)
    near : float
        the distance of the near clipping plane
    far : float
        the distance of the far clipping plane
    device : str or torch.device (optional)
        the device the tensor will be stored to (default is 'cuda:0')

    Returns
    -------
    Tensor
        a (4,4,) projection matrix
    """

    M       = eye(4, device=device)
    M[0, 0] = 1/(aspect*tan(fov/2))
    M[1, 1] = 1/tan(fov/2)
    M[2, 2] = -(far+near)/(far-near)
    M[2, 3] = -2 * (far * near) / (far-near)
    M[3, 2] = -1
    return M


def bokeh_camera(P, n=4, aperture=PI_16):
    """
    Creates a set of n positions around the given one

    Parameters
    ----------
    P : Tensor
        a (1,3,) tensor
    n : int (optional)
        the number of positions to generate (default is 4)
    aperture : float (optional)
        the aperture angle in radians (default is PI/16)

    Returns
    -------
    Tensor
        a (N+1,3,) tensor
    """

    Q = torch.cat((torch.zeros_like(P), aperture*equilateral_polygon(n, device=P.device)), dim=0)
    return sph2cart(cart2sph(P)+Q[:, (2, 0, 1)])


def camera_stage(tile=(6, 4), camera_distance=1, to_spherical=False, device='cuda:0'):
    """
    Returns the positions of a camera lying on the vertices of a sphere with given tiling

    Parameters
    ----------
    tile : tuple (optional)
        the sphere elevation-azimuth tiling (default is (6,4))
    camera_distance : float (optional)
        the camera distance from the origin (default is 1)
    to_spherical : bool (optional)
        if True, converts the coordinates into spherical (default is False)
    device : str or torch.device
        the device the tensors will be stored to (default is 'cuda:0')

    Returns
    -------
    (Tensor,LongTensor)
        the positions and the edge tensor of the camera views
    """

    P, T  = Sphere(tile=tile, device=device)[0:2]
    P,T   = soup2mesh(P, T)[0:2]
    theta = PI/100
    R     = torch.tensor([[1, 0, 0],
                          [0, cos(theta), -sin(theta)],
                          [0, sin(theta), cos(theta)]], dtype=torch.float, device=device)
    P     = torch.mul(torch.mm(P, torch.t(R)), camera_distance)
    E = torch.cat((poly2edge(T)[0],
                   poly2edge(torch.cat((ind2edge(T[0], T[2]),
                                        ind2edge(T[1], T[3])), dim=1))[0]),
                  dim=1)
    E = poly2unique(E[:, E[0] != E[1]], winding=True)[0]
    if to_spherical:
        P = cart2sph(P)
    return P, E


def camera_from_polyhedron(polyhedronFcn, camera_distance=1, to_spherical=False, device='cuda:0'):
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
        the positions and the edge tensor of the camera views
    """

    P, T  = polyhedronFcn(device=device)[0:2]
    theta = PI/100
    R     = torch.tensor([[1, 0, 0],
                          [0, cos(theta), -sin(theta)],
                          [0, sin(theta), cos(theta)]], dtype=torch.float, device=device)
    P     = torch.mul(torch.mm(normr(P), torch.t(R)), camera_distance)
    if to_spherical:
        P = cart2sph(P)
    return P, poly2edge(T)[0]


def camera_6(camera_distance=1, to_spherical=False, device='cuda:0'):
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
        the positions and the edge tensor of the camera views
    """

    return camera_from_polyhedron(Octahedron, camera_distance=camera_distance, to_spherical=to_spherical, device=device)


def camera_12(camera_distance=1, to_spherical=False, device='cuda:0'):
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
        the positions and the edge tensor of the camera views
    """

    return camera_from_polyhedron(Icosahedron, camera_distance=camera_distance, to_spherical=to_spherical, device=device)


def camera_18(camera_distance=1, to_spherical=False, device='cuda:0'):
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
        the positions and the edge tensor of the camera views
    """

    return camera_from_polyhedron(Octahedron_2, camera_distance=camera_distance, to_spherical=to_spherical, device=device)


def camera_42(camera_distance=1, to_spherical=False, device='cuda:0'):
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
        the positions and the edge tensor of the camera views
    """

    return camera_from_polyhedron(Icosahedron_2, camera_distance=camera_distance, to_spherical=to_spherical, device=device)


def camera_66(camera_distance=1, to_spherical=False, device='cuda:0'):
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
        the positions and the edge tensor of the camera views
    """

    return camera_from_polyhedron(Octahedron_3, camera_distance=camera_distance, to_spherical=to_spherical, device=device)


def camera_n(n, camera_distance=1, to_spherical=False, device='cuda:0'):
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

    P = torch.mul(equilateral_polygon(n, device=device), camera_distance)
    E = torch.t(torch.cat((indices(0, n-2, device=device), indices(1, n-1, device=device)), dim=1))
    if to_spherical:
        P = cart2sph(P)
    return P, poly2edge(E)[0]
