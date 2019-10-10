import torch
from ..math.tan                     import *
from ..math.cart2affine             import *
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


class CameraExtrinsic(object):
    def __init__(self, position=(0, 0, -1), target=(0, 0, 0), up_vector=(0, 1, 0), device='cuda:0'):
        self.position  = torch.tensor(position,  dtype=torch.float, device=device)
        self.target    = torch.tensor(target,    dtype=torch.float, device=device)
        self.up_vector = torch.tensor(up_vector, dtype=torch.float, device=device)
        self.device    = device

    def look_at(self, target):
        self.target = target
        return self

    def look_from(self, position):
        self.position = position
        return self

    def direction(self):
        return self.target - self.position

    def view_matrix(self):
        """
        Returns the view matrix

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
        if 'device' in kwargs:
            self.device = kwargs['device']
        self.position  = self.position.to(**kwargs)
        self.target    = self.target.to(**kwargs)
        self.up_vector = self.up_vector.to(**kwargs)
        return self

    def __setattr__(self, key, value):
        self.__dict__[key] = value
        if key == 'device':
            self.position = self.position.to(self.device)
            self.target = self.target.to(self.device)
            self.up_vector = self.up_vector.to(self.device)


class CameraIntrinsic(object):
    def __init__(self, fov=30, near=0.1, far=10, image_size=(256, 256), projection='perspective', device='cuda:0'):
        self.fov        = fov
        self.near       = near
        self.far        = far
        self.image_size = image_size
        self.projection = projection
        self.device     = device

    def aspect(self):
        return self.image_size[0] / self.image_size[1]

    def projection_matrix(self):
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

        M = torch.zeros(4, 4, device=self.device)
        M[0, 0] = 1 / (self.aspect() * tan(self.fov / 2))
        M[1, 1] = 1 / tan(self.fov / 2)
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

        M = torch.zeros(4, 4, device=self.device)
        M[0, 0] = 1 / (self.aspect() * tan(self.fov / 2))
        M[1, 1] = 1 / tan(self.fov / 2)
        M[2, 2] = (self.far + self.near) / (self.far - self.near)
        M[2, 3] = -2 * (self.far * self.near) / (self.far - self.near)
        M[3, 2] = 1
        return M

    def to(self, **kwargs):
        if 'device' in kwargs:
            self.device = kwargs['device']
        return self


class Camera(object):
    def __init__(self,
                 extrinsic=CameraExtrinsic(),
                 intrinsic=CameraIntrinsic(),
                 name='Camera', device='cuda:0'):
        self.extrinsic = extrinsic
        self.intrinsic = intrinsic
        self.name      = name
        self.device    = device

    def project(self, P):
        w = self.intrinsic.image_size[0]
        h = self.intrinsic.image_size[1]
        v = torch.tensor([[w/2, h/2, 1/2]], dtype=torch.float, device=P.device)
        Q = torch.matmul(cart2affine(P, w=1),
                         torch.matmul(self.intrinsic.projection_matrix(),
                                      self.extrinsic.view_matrix()).t())
        return (Q[:, :3] / Q[:, 3]) * v + v

    def unproject(self, Q):
        w = self.intrinsic.image_size[0]
        h = self.intrinsic.image_size[1]
        v = torch.tensor([[2/w, 2/h, 2]], dtype=torch.float, device=Q.device)
        P = torch.matmul(cart2affine(Q*v-1, w=1),
                         torch.inverse(torch.matmul(self.intrinsic.projection_matrix(),
                                                    self.extrinsic.view_matrix())).t())
        return P[:, :3] / (1 / P[:, 3])

    def to(self, **kwargs):
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
