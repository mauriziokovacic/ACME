import torch
from .degree import *


def laplacian(A):
    """
    Returns the laplacian matrix from a given adjacency matrix

    Parameters
    ----------
    A : Tensor
        an adjacency matrix

    Returns
    -------
    Tensor
        the laplacian matrix
    """

    return degree(A)-A


def symmetric_normalized_laplacian(A):
    """
    Returns the symmetric normalized laplacian matrix from a given adjacency matrix

    Parameters
    ----------
    A : Tensor
        an adjacency matrix

    Returns
    -------
    Tensor
        the laplacian matrix
    """

    D = torch.diag(torch.reciprocal(torch.sqrt(torch.diag(degree(A)))))
    return torch.mm(D, torch.mm(laplacian(A), D))


def random_walk_normalized_laplacian(A):
    """
    Returns the random walk normalized laplacian matrix from a given adjacency matrix

    Parameters
    ----------
    A : Tensor
        an adjacency matrix

    Returns
    -------
    Tensor
        the laplacian matrix
    """

    D = torch.diag(torch.reciprocal(torch.diag(degree(A))))
    return torch.mm(D, laplacian(A))
