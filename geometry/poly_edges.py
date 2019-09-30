from ..math.norm import *


def poly_edges(P, T):
    """
    Returns the ordered edges from the given polygons

    Parameters
    ----------
    P : Tensor
        a (N, D,) points set tensor
    T : LongTensor
        a (M, T,) topology tensor

    Returns
    -------
    tuple
        a tuple containing the edges of the given polygons
    """

    p = P[torch.cat((T, T[0].unsqueeze(0)), dim=0)]
    return tuple(p[1:]-p[:-1])


def poly_edges_length(P, T):
    """
    Returns the per polygon edge lengths

    Parameters
    ----------
    P : Tensor
        a (N, D,) points set tensor
    T : LongTensor
        a (M, T,) topology tensor

    Returns
    -------
    Tensor
        the (T, M,) edge lengths tensor
    """

    E = poly_edges(P, T)
    return torch.cat([norm(e) for e in E], dim=1)


def poly_edges_min_length(P, T):
    """
    Returns the per polygon min edge length

    Parameters
    ----------
    P : Tensor
        a (N, D,) points set tensor
    T : LongTensor
        a (M, T,) topology tensor

    Returns
    -------
    Tensor
        the (T, M,) min edge length tensor
    """

    return torch.min(poly_edges_length(P, T), dim=1, keepdim=True)[0]


def poly_edges_max_length(P, T):
    """
    Returns the per polygon max edge length

    Parameters
    ----------
    P : Tensor
        a (N, D,) points set tensor
    T : LongTensor
        a (M, T,) topology tensor

    Returns
    -------
    Tensor
        the (T, M,) max edge length tensor
    """

    return torch.max(poly_edges_length(P, T), dim=1, keepdim=True)[0]


def poly_edges_mean_length(P, T):
    """
    Returns the mean edge length of the input model

    Parameters
    ----------
    P : Tensor
        a (N, D,) points set tensor
    T : LongTensor
        a (M, T,) topology tensor

    Returns
    -------
    Tensor
        the (1,) mean edge length
    """

    return torch.mean(poly_edges_length(P, T))


def poly_edges_median_length(P, T):
    """
    Returns the median edge length of the input model

    Parameters
    ----------
    P : Tensor
        a (N, D,) points set tensor
    T : LongTensor
        a (M, T,) topology tensor

    Returns
    -------
    Tensor
        the (1,) median edge length tensor
    """

    return torch.median(poly_edges_length(P, T))
