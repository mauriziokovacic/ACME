import torch


def sph2cart(P):
    """
    Converts the input spherical coordinates [r,theta,phi] into cartesian coordinates [x,y,z]

    In spherical coordinates, r represents the radius, theta the elevetion and phi the azimuth

    Parameters
    ----------
    P : Tensor
        a (N,3,) tensor containing [r,theta,phi]

    Returns
    -------
    Tensor
        a (N,3,) tensor containing [x,y,z]
    """

    r, theta, phi = torch.t(P) #radius elevation azimuth
    r            = r.unsqueeze(1)
    theta        = theta.unsqueeze(1)
    phi          = phi.unsqueeze(1)
    z            = r * torch.sin(theta)
    rcoselev     = r * torch.cos(theta)
    x            = rcoselev * torch.cos(phi)
    y            = rcoselev * torch.sin(phi)
    return torch.cat((x, y, z), dim=1)
