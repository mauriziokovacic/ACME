import torch
from ACME.utility.row          import *
from ACME.utility.nop          import *
from ACME.utility.indices      import *
from ACME.math.constant        import *
from ACME.math.cos             import *
from ACME.math.sin             import *
from ACME.math.normvec         import *
from ACME.math.cart2sph        import *
from ACME.math.sph2rotm        import *
from ACME.topology.ind2poly    import *
from ACME.topology.poly2poly   import *
from ACME.topology.poly2unique import *
from ACME.geometry.octahedron  import *
from ACME.geometry.icosahedron import *
from ACME.geometry.shape_scale import *
from ACME.geometry.sphere      import *
from ACME.geometry.soup2mesh   import *



def camera_stage(tile=(6,4),camera_distance=1,to_spherical=False,device='cuda:0'):
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

    P,T   = Sphere(tile=tile,device=device)[0:2]
    P,T   = soup2mesh(P,T)[0:2]
    theta = PI/100
    R     = torch.tensor([[1,0,0],[0,cos(theta),-sin(theta)],[0,sin(theta),cos(theta)]],dtype=torch.float,device=device)
    P     = torch.mul(torch.mm(P,torch.t(R)),camera_distance)
    E = torch.cat((poly2edge(T)[0],
                   poly2edge(torch.cat((ind2edge(T[0],T[2]),
                                        ind2edge(T[1],T[3])),dim=1))[0]),
                   dim=1)
    E = poly2unique(E[:,E[0]!=E[1]],winding=True)
    if to_spherical:
        P = cart2sph(P)
    return P,E



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
        the positions and the edge tensor of the camera views
    """

    P,T   = hedronFcn(device=device)[0:2]
    theta = PI/100
    R     = torch.tensor([[1,0,0],[0,cos(theta),-sin(theta)],[0,sin(theta),cos(theta)]],dtype=torch.float,device=device)
    P     = torch.mul(torch.mm(normr(P),torch.t(R)),camera_distance)
    if to_spherical:
        P = cart2sph(P)
    return P,poly2edge(T)[0]



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
        the positions and the edge tensor of the camera views
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
        the positions and the edge tensor of the camera views
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
        the positions and the edge tensor of the camera views
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
        the positions and the edge tensor of the camera views
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
        the positions and the edge tensor of the camera views
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
    E = torch.t(torch.cat((indices(0,n-2,device=device),indices(1,n-1,device=device)),dim=1))
    if to_spherical:
        P = cart2sph(P)
    return P,poly2edge(E)[0]
