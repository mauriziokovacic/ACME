import torch
from math.dot import *


def blend_transform(T,W):
    """
    Blends the given flattened 3x4 transforms with the specified weights

    Transforms are intended being 3D affine transforms in the form:
        |R t|
        |0 1|
    where the last row is taken out. The transforms needs to be flattened.

    Parameters
    ----------
    T : Tensor
        a (H,12,) tensor representing a 3x4 matrix
    W : Tensor
        the (N,H,) weights matrix

    Returns
    -------
    Tensor
        a (N,12,) tensor representing the transform of every vertex
    """

    return torch.mm(W,T)



def transform_point(T,P):
    """
    Transforms the given points with the specified transforms

    Parameters
    ----------
    T : Tensor
        a (N,12,) tensor representing the transforms
    P : Tensor
        a (N,3,) points set tensor

    Returns
    -------
    Tensor
        a (N,3,) points set tensor
    """

    return torch.cat((dot(T[:,0: 3],P)+T[:, 3],
                      dot(T[:,4: 7],P)+T[:, 7],
                      dot(T[:,8:11],P)+T[:,11]),dim=1)



def transform_normal(T,N):
    """
    Transforms the given normals with the specified transforms

    Parameters
    ----------
    T : Tensor
        a (N,12,) tensor representing the transforms
    P : Tensor
        a (N,3,) normals set tensor

    Returns
    -------
    Tensor
        a (N,3,) normals set tensor
    """

    return torch.cat((dot(T[:,0: 3],N),
                      dot(T[:,4: 7],N),
                      dot(T[:,8:11],N)),dim=1)



def linear_blend_skinning(P,N,T,W):
    """
    Performs the linear blend skinning

    Parameters
    ----------
    P : Tensor
        the (N,3,) points set tensor
    N : Tensor
        the (N,3,) normals set tensor
    T : Tensor
        the (H,12,) transforms tensor
    W : Tensor
        the (N,H,) weights matrix

    Returns
    -------
    (Tensor, Tensor)
        the transformed points and normals sets tensors
    """

    t = blend_transform(T,W)
    return transform_point(P,t), normr(transform_normal(N,t))

