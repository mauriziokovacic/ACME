import numpy
from ..utility.col         import *
from ..utility.flatten     import *
from ..utility.transpose   import *
from ..utility.repelem     import *
from ..utility.torch2numpy import *
from .ispoly import *


def _xmesh(T, scheme, iter=None):
    """
    Creates the subdivision matrix of the given topology tensor w.r.t. the input scheme.

    Parameters
    ----------
    T : LongTensor
        the topology tensor
    scheme : (numpy.ndarray, int)
        the subdivision scheme with the resulting polygons sides
    iter : int (optional)
        the number of subdivision iteration to perform. No iteration if None (default is None)

    Returns
    -------
    (Tensor, LongTensor)
        the subdivision matrix and the new topology tensor
    """

    def idx(s, e):
        return numpy.expand_dims(numpy.arange(s, e), axis=1)

    def repel(a, dim):
        out = a
        for d in range(0, len(dim)):
            out = numpy.repeat(out, dim[d], axis=d)
        return out

    t = torch.t(T).cpu().numpy()
    M = numpy.identity(numpy.max(t) + 1)
    if iter is not None:
        t = flatten(t)
        M = M[t]
        for i in range(0, iter):
            e = repel(idx(0, row(M)//row(T)), (row(scheme[0]), 1)) * row(T) +\
                numpy.tile(scheme[0], (row(M) // row(T), 1))
            m = numpy.zeros((row(e), col(M)))
            for j in range(0, col(e)):
                m += M[e[:, j]]
            M = (1 / col(e)) * m
        t = transpose(numpy.reshape(idx(0, row(M)), (row(M) // scheme[1], scheme[1])))
    return numpy2torch(M, dtype=torch.float, device=T.device), \
           numpy2torch(t, dtype=torch.long, device=T.device)


def xedge(T, iter=None):
    """
    Creates the naive subdivision matrix for a triangle topology.

    Parameters
    ----------
    T : LongTensor
        the (2,N,) topology tensor
    iter : int (optional)
        the number of iteration to perform. No iteration if None (default is None)

    Returns
    -------
    (Tensor,LongTensor)
        the subdivision matrix and the new (2,M,) topology tensor

    Raises
    ------
    AssertionError
        if topology is not a valid edge topology
    """

    assert isedge(T), 'Input topology must be an edge topology'

    i = 0
    j = 1
    scheme = (numpy.array([[i, i], [i, j],
                           [j, i], [j, j],]), 2)
    return _xmesh(T, scheme, iter=iter)


def xtri(T, iter=None):
    """
    Creates the naive subdivision matrix for a triangle topology.

    Parameters
    ----------
    T : LongTensor
        the (3,N,) topology tensor
    iter : int (optional)
        the number of iteration to perform. No iteration if None (default is None)

    Returns
    -------
    (Tensor,LongTensor)
        the subdivision matrix and the new (3,M,) topology tensor

    Raises
    ------
    AssertionError
        if topology is not a valid triangle topology
    """

    assert istri(T), 'Input topology must be a triangle topology'

    i = 0
    j = 1
    k = 2
    scheme = (numpy.array([[i, i], [i, j], [i, k],
                           [j, j], [j, k], [j, i],
                           [k, k], [k, i], [k, j],
                           [i, j], [j, k], [k, i]]), 3)
    return _xmesh(T, scheme, iter=iter)


def xquad(T, iter=None):
    """
    Creates the naive subdivision matrix for a quad topology.

    Parameters
    ----------
    T : LongTensor
        the (4,N,) topology tensor
    iter : int (optional)
        the number of iteration to perform. No iteration if None (default is None)

    Returns
    -------
    (Tensor,LongTensor)
        the subdivision matrix and the new (4,M,) topology tensor

    Raises
    ------
    AssertionError
        if topology is not a valid quad topology
    """

    assert isquad(T), 'Input topology must be a quad topology'

    i = 0
    j = 1
    k = 2
    l = 3
    scheme = (numpy.array([[i, i, i, i], [i, i, j, j], [i, j, k, l], [i, i, l, l],
                           [j, j, j, j], [j, j, k, k], [i, j, k, l], [j, j, i, i],
                           [k, k, k, k], [k, k, l, l], [i, j, k, l], [k, k, j, j],
                           [l, l, l, l], [l, l, i, i], [i, j, k, l], [l, l, k, k]]), 4)
    return _xmesh(T, scheme, iter=iter)


def xtri2quad(T, *args, **kwargs):
    """
    Creates the subdivision matrix for transforming a triangle topology into a quad one

    Parameters
    ----------
    T : LongTensor
        the (3,N,) topology tensor

    Returns
    -------
    (Tensor,LongTensor)
        the subdivision matrix and the new (4,M,) topology tensor

    Raises
    ------
    AssertionError
        if topology is not a valid triangle topology
    """

    assert istri(T), 'Input topology must be a triangle topology'

    i = 0
    j = 1
    k = 2
    scheme = (numpy.array([[i, i, i, i, i, i], [i, i, i, j, j, j], [i, i, j, j, k, k], [i, i, i, k, k, k],
                           [j, j, j, j, j, j], [j, j, j, k, k, k], [i, i, j, j, k, k], [j, j, j, i, i, i],
                           [k, k, k, k, k, k], [k, k, k, i, i, i], [i, i, j, j, k, k], [k, k, k, j, j, j]]), 4)
    iter = 1
    return _xmesh(T, scheme, iter)


def xtet(T, iter=None):
    """
    Creates the naive subdivision matrix for a tetrahedra topology.

    Parameters
    ----------
    T : LongTensor
        the (4,N,) topology tensor
    iter : int (optional)
        the number of iteration to perform. No iteration if None (default is None)

    Returns
    -------
    (Tensor,LongTensor)
        the subdivision matrix and the new (4,M,) topology tensor

    Raises
    ------
    AssertionError
        if topology is not a valid tetrahedra topology
    """

    assert istet(T), 'Input topology must be a tetrahedra topology'

    i = 0
    j = 1
    k = 2
    l = 3
    scheme = (numpy.array([[i, i], [i, j], [i, k], [i, l],
                           [j, j], [j, l], [j, k], [j, i],
                           [k, k], [k, i], [k, j], [k, l],
                           [l, l], [l, i], [l, k], [l, j],

                           [i, k], [i, l], [i, j], [k, l],
                           [i, k], [i, j], [j, k], [k, l],
                           [j, l], [i, j], [i, l], [k, l],
                           [j, l], [j, k], [i, j], [k, l],
                           ]), 4)
    return _xmesh(T, scheme=scheme, iter=iter)


def xhex(T, iter=None):
    """
    Creates the naive subdivision matrix for a hexaedra topology.

    Parameters
    ----------
    T : LongTensor
        the (8,N,) topology tensor
    iter : int (optional)
        the number of iteration to perform. No iteration if None (default is None)

    Returns
    -------
    (Tensor,LongTensor)
        the subdivision matrix and the new (8,M,) topology tensor

    Raises
    ------
    AssertionError
        if topology is not a valid hexaedra topology
    """

    assert ishex(T), 'Input topology must be a hexaedra topology'

    i = 0
    j = 1
    k = 2
    l = 3
    a = 4
    b = 5
    c = 6
    d = 7

    scheme = (numpy.array([[i, i, i, i, i, i, i, i], [i, i, i, i, j, j, j, j],
                           [i, i, j, j, k, k, l, l], [i, i, i, i, l, l, l, l],
                           [i, i, i, i, a, a, a, a], [i, i, j, j, a, a, b, b],
                           [i, j, k, l, a, b, c, d], [i, i, l, l, a, a, d, d],

                           [j, j, j, j, j, j, j, j], [j, j, j, j, b, b, b, b],
                           [j, j, b, b, c, c, k, k], [j, j, j, j, k, k, k, k],
                           [j, j, j, j, i, i, i, i], [i, i, j, j, a, a, b, b],
                           [i, j, k, l, a, b, c, d], [i, i, j, j, k, k, l, l],

                           [k, k, k, k, k, k, k, k], [k, k, k, k, c, c, c, c],
                           [k, k, l, l, c, c, d, d], [k, k, k, k, l, l, l, l],
                           [k, k, k, k, j, j, j, j], [j, j, k, k, b, b, c, c],
                           [i, j, k, l, a, b, c, d], [i, i, j, j, k, k, l, l],

                           [l, l, l, l, l, l, l, l], [l, l, l, l, k, k, k, k],
                           [k, k, l, l, c, c, d, d], [l, l, l, l, d, d, d, d],
                           [l, l, l, l, i, i, i, i], [i, i, j, j, k, k, l, l],
                           [i, j, k, l, a, b, c, d], [i, i, l, l, a, a, d, d],

                           [a, a, a, a, a, a, a, a], [a, a, a, a, i, i, i, i],
                           [i, i, l, l, a, a, d, d], [a, a, a, a, d, d, d, d],
                           [a, a, a, a, b, b, b, b], [i, i, j, j, a, a, b, b],
                           [i, j, k, l, a, b, c, d], [a, a, b, b, c, c, d, d],

                           [b, b, b, b, b, b, b, b], [a, a, a, a, b, b, b, b],
                           [a, a, b, b, c, c, d, d], [b, b, b, b, c, c, c, c],
                           [b, b, b, b, j, j, j, j], [i, i, j, j, a, a, b, b],
                           [i, j, k, l, a, b, c, d], [j, j, k, k, b, b, c, c],

                           [c, c, c, c, c, c, c, c], [c, c, c, c, d, d, d, d],
                           [k, k, l, l, c, c, d, d], [c, c, c, c, k, k, k, k],
                           [c, c, c, c, b, b, b, b], [a, a, b, b, c, c, d, d],
                           [i, j, k, l, a, b, c, d], [j, j, k, k, b, b, c, c],

                           [d, d, d, d, d, d, d, d], [d, d, d, d, l, l, l, l],
                           [k, k, l, l, c, c, d, d], [d, d, d, d, c, c, c, c],
                           [a, a, a, a, d, d, d, d], [i, i, l, l, a, a, d, d],
                           [i, j, k, l, a, b, c, d], [a, a, b, b, c, c, d, d]]),
              8)

    return _xmesh(T, scheme=scheme, iter=iter)


def xtet2hex(T, *args, **kwargs):
    """
    Creates the subdivision matrix for transforming a tetrahedra topology into a hexaedra one

    Parameters
    ----------
    T : LongTensor
        the (4,N,) topology tensor

    Returns
    -------
    (Tensor,LongTensor)
        the subdivision matrix and the new (8,M,) topology tensor

    Raises
    ------
    AssertionError
        if topology is not a valid tetrahedra topology
    """

    assert istet(T), 'Input topology must be a tetrahedra topology'

    i = 0
    j = 1
    k = 2
    l = 3
    scheme = (numpy.array([[i, i, i, i, i, i, i, i, i, i, i, i], [i, i, i, i, i, i, k, k, k, k, k, k],
                           [i, i, i, i, j, j, j, j, k, k, k, k], [i, i, i, i, i, i, j, j, j, j, j, j],
                           [i, i, i, i, i, i, l, l, l, l, l, l], [i, i, i, i, k, k, k, k, l, l, l, l],
                           [i, i, i, j, j, j, k, k, k, l, l, l], [i, i, i, i, j, j, j, j, l, l, l, l],

                           [j, j, j, j, j, j, j, j, j, j, j, j], [j, j, j, j, j, j, k, k, k, k, k, k],
                           [j, j, j, j, k, k, k, k, l, l, l, l], [j, j, j, j, j, j, l, l, l, l, l, l],
                           [i, i, i, i, i, i, j, j, j, j, j, j], [i, i, i, i, j, j, j, j, k, k, k, k],
                           [i, i, i, j, j, j, k, k, k, l, l, l], [i, i, i, i, j, j, j, j, l, l, l, l],

                           [k, k, k, k, k, k, k, k, k, k, k, k], [j, j, j, j, j, j, k, k, k, k, k, k],
                           [i, i, i, i, j, j, j, j, k, k, k, k], [k, k, k, k, k, k, i, i, i, i, i, i],
                           [k, k, k, k, k, k, l, l, l, l, l, l], [j, j, j, j, k, k, k, k, l, l, l, l],
                           [i, i, i, j, j, j, k, k, k, l, l, l], [i, i, i, i, k, k, k, k, l, l, l, l],

                           [l, l, l, l, l, l, l, l, l, l, l, l], [k, k, k, k, k, k, l, l, l, l, l, l],
                           [i, i, i, i, k, k, k, k, l, l, l, l], [i, i, i, i, i, i, l, l, l, l, l, l],
                           [j, j, j, j, j, j, l, l, l, l, l, l], [j, j, j, j, k, k, k, k, l, l, l, l],
                           [i, i, i, j, j, j, k, k, k, l, l, l], [j, j, j, j, i, i, i, i, l, l, l, l], ]), 8)
    iter = 1
    return _xmesh(T, scheme, iter)
