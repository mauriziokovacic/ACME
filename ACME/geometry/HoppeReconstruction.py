from ..math.AbstractImplicitSurface import *
from ..math.knn                     import *
from ..math.unrooted_norm           import *
from .point_plane_distance          import *


class HoppeReconstruction(AbstractImplicitSurface):
    def __init__(self, points, normals):
        super(HoppeReconstruction, self).__init__()
        self.p = points
        self.n = normals

    def f(self, x):
        i = knn(x, self.p, k=1, distFcn=sqdistance)[0]
        return point_plane_distance(self.p[i], self.n[i], x)

    def df(self, x):
        i = knn(x, self.p, k=1, distFcn=sqdistance)[0]
        return self.n[i]