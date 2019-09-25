from ..utility.col        import *
from ..utility.bi2de      import *
from ..utility.find       import *
from ..utility.LongTensor import *
from ..utility.isempty    import *
from ..utility.indices    import *
from ..utility.sum        import *
from ..topology.poly2ind  import *
from ..topology.ispoly    import *


class Segment(object):
    def __init__(self, A=None, B=None):
        self.A = A
        self.B = B


class SurfaceSegment(object):
    def __init__(self, T=None, **kwargs):
        super(SurfaceSegment, self).__init__(**kwargs)
        self.T = T


def meandering_triangle(T, F, iso_target=0.5):
    """
    Returns a collection of segments defined on a surface

    Parameters
    ----------
    T : LongTensor
        the triangle topology tensor
    F : Tensor
        a (N,1,) tensor defining a scalar field on the nodes of the model
    iso_target : float (optional)
        a iso value to track on the surface (default is 0.5)

    Returns
    -------
    list or SurfaceSegment
        a collection of segments defined on the surface of a model
    """

    if not istri(T):
        raise RuntimeError('Topology tensor must be triangles')
    table = LongTensor([[0, 0, 0],
                        [1, 3, 2],
                        [2, 1, 3],
                        [3, 1, 2],
                        [3, 2, 1],
                        [2, 3, 1],
                        [1, 2, 3],
                        [0, 0, 0]], device=T.device)
    code    = lambda v: bi2de(v >= iso_target)
    fetch   = lambda v: table[code(v), :].squeeze(1)
    I, J, K = tri2ind(T)
    C = [[] for i in range(col(F))]
    for f in range(col(F)):
        Fi = torch.cat((F[I, f].unsqueeze(1),
                        F[J, f].unsqueeze(1),
                        F[K, f].unsqueeze(1)), dim=1)
        X  = fetch(Fi)
        n  = find(X[:, 1])
        if not isempty(n):
            X  -= 1
            s   = (numel(n), 3)
            c   = SurfaceSegment()
            c.A = torch.zeros(s, dtype=torch.float, device=T.device)
            c.B = torch.zeros(s, dtype=torch.float, device=T.device)
            c.T = n
            t   = torch.zeros((row(X), 2), dtype=torch.float, device=T.device)
            t[n, :] = torch.cat((((iso_target - Fi[n, X[n, 1]]) / (Fi[n, X[n, 0]] - Fi[n, X[n, 1]])).unsqueeze(1),
                                 ((iso_target - Fi[n, X[n, 2]]) / (Fi[n, X[n, 0]] - Fi[n, X[n, 2]])).unsqueeze(1)), dim=1)

            i = indices(0, numel(n) - 1, device=T.device)
            c.A[i, X[n, 0]] =     t[n, 0]
            c.A[i, X[n, 1]] = 1 - t[n, 0]
            c.A[c.A < 0.01] = 0
            c.A = c.A / sum(c.A, 1)

            c.B[i, X[n, 0]] =     t[n, 1]
            c.B[i, X[n, 2]] = 1 - t[n, 1]
            c.B[c.B < 0.01] = 0
            c.B = c.B / sum(c.B, 1)

            i = find(sum((c.A - c.B)**2, 1) > 0.0001)
            c.A = c.A[i, :]
            c.B = c.B[i, :]
            c.T = c.T[i]
            C[f] = c
    return C if len(C)>0 else C[0]