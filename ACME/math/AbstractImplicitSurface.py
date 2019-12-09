from ..utility.ACMEClass import *


class AbstractImplicitSurface(ACMEClass):
    def __init__(self):
        if isinstance(self, AbstractImplicitSurface):
            raise RuntimeError('Cannot instantiate an abstract implicit surface')
        super(AbstractImplicitSurface, self).__init__()

    def f(self, x):
        raise NotImplementedError

    def df(self, x):
        raise NotImplementedError

    def __call__(self, x, *args, **kwargs):
        return self.f(x, *args, **kwargs)