from ..math.AbstractImplicitSurface  import *
from ..math.prepare_broadcast        import *
from ..math.norm                     import *
from ..math.triharmonic_rbf          import *
from ..math.wendland_rbf             import *
from ..linear_problem.linear_problem import *
from ..utility.ConstantTensor        import *


class RBFReconstruction(AbstractImplicitSurface):
    def __init__(self, centers, normals, rbf, eps=0.0001):
        super(RBFReconstruction, self).__init__()
        self.c   = centers
        self.rbf = rbf
        self.w   = self.__weights__(normals, eps=eps)

    def __weights__(self, normals, eps=0.0001):
        x,  c  = prepare_broadcast(self.c, self.c)
        xn, cn = prepare_broadcast(self.c + eps * normals, self.c)
        d      = torch.cat((torch.zeros(x.shape[0], dtype=x.dtype, device=x.device),
                            ConstantTensor(eps, dtype=xn.dtype, device=xn.device)))
        A = self.rbf(torch.cat((distance(x, c).squeeze(), distance(xn, cn).squeeze()), dim=0))
        w = linear_problem(A, d, eps=0).unsqueeze(1)
        return w

    def f(self, x):
        y, c = prepare_broadcast(x, self.c)
        return torch.matmul(self.rbf(distance(y, c)), self.w)

    @staticmethod
    def WendlandReconstruction(center, normals, eps=0.0001):
        return RBFReconstruction(centers=center, normals=normals, rbf=Wendland_rbf, eps=eps)

    @staticmethod
    def TriharmonicReconstruction(center, normals, eps=0.0001):
        return RBFReconstruction(centers=center, normals=normals, rbf=triharmonic_rbf, eps=eps)
