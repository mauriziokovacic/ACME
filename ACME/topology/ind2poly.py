import torch
from ..utility.to_row import *


def ind2poly(*I):
    """
    Creates polygons from the input indices

    Parameters
    ----------
    *I : LongTensors...
        a sequence of LongTensors

    Returns
    -------
    LongTensor
        the topology tensor representing the polygons

    Raises
    ------
    AssertionError
        if inputs are less than two
    """

    assert len(I) >= 2, 'Input must be at least two tensors'
    return torch.cat(tuple(to_row(*I)), dim=0)


def ind2edge(I, J):
    """
    Creates the 2xn edge tensor from the inputs

    Parameters
    ----------
    I : LongTensor
        a 1D tensor
    J : LongTensor
        a 1D tensor

    Returns
    -------
    LongTensor
        the 2xn edge tensor
    """

    return ind2poly(I, J)


def ind2tri(I, J, K):
    """
    Creates the 3xn triangle tensor from the inputs

    Parameters
    ----------
    I : LongTensor
        a 1D tensor
    J : LongTensor
        a 1D tensor
    K : LongTensor
        a 1D tensor

    Returns
    -------
    LongTensor
        the 3xn triangle tensor
    """

    return ind2poly(I, J, K)


def ind2quad(I, J, K, L):
    """
    Creates the 4xn quad tensor from the inputs

    Parameters
    ----------
    I : LongTensor
        a 1D tensor
    J : LongTensor
        a 1D tensor
    K : LongTensor
        a 1D tensor
    L : LongTensor
        a 1D tensor

    Returns
    -------
    LongTensor
        the 4xn quad tensor
    """

    return ind2poly(I, J, K, L)
