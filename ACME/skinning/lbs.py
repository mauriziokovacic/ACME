from ..math.normvec   import *
from ..math.cart2homo import *
from ..utility.matmul import *


def blend_transform(W, T):
    """
    Blends the given transforms with the specified weights

    Parameters
    ----------
    W : Tensor
        the (N,H,) weights matrix
    T : Tensor
        a (H,R,C,) tensor representing the transforms

    Returns
    -------
    Tensor
        a (N,R,C,) tensor representing the transform of every vertex
    """

    return matmul(W, T.view(T.size(0), -1)).view(-1, *T.size()[1:])


def transform(T, X, mode='point'):
    """
    Transforms the given set with the specified affine transforms

    Parameters
    ----------
    T : Tensor
        a (N,R,C) tensor representing the transforms
    X : Tensor
        a (N,3,) tensor
    mode : str (optional)
        either 'point' or 'normal', depending on the argument

    Returns
    -------
    Tensor
        a (N,3,) points set tensor
    """

    if strcmpi(mode, 'point'):
        w = 1
    else:
        if strcmpi(mode, 'normal'):
            w = 0
    return matmul(T,
                  cart2homo(X, w=w).view(-1, X.size(1) + 1, 1)
                  )[:, :-1].view(*X.size())


def transform_point(T, P):
    """
    Transforms the given normals with the specified transforms

    Parameters
    ----------
    T : Tensor
        a (N,12,) tensor representing the transforms
    P : Tensor
        a (N,3,) points set tensor

    Returns
    -------
    Tensor
        a (N,3,) normals set tensor
    """

    return transform(T, P, mode='point')


def transform_normal(T, N):
    """
    Transforms the given normals with the specified transforms

    Parameters
    ----------
    T : Tensor
        a (N,12,) tensor representing the transforms
    N : Tensor
        a (N,3,) normals set tensor

    Returns
    -------
    Tensor
        a (N,3,) normals set tensor
    """

    return transform(T, N, mode='normal')


def linear_blend_skinning(P, N, T, W):
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

    t = blend_transform(T, W)
    return transform_point(P, t), normr(transform_normal(N, t))

