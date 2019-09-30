from ..utility.col   import *
from .normw          import *
from .limit_weight   import *


class Skin(object):
    def __init__(self, mesh=None, weight=None, name='Skin'):
        self.mesh     = mesh
        self.weight   = weight
        self.__weight = weight.clone()
        self.name     = name

    def num_handles(self):
        return col(self.weight)

    def normalize(self):
        self.weight = normw(self.__weight)
        return self

    def limit_weight(self, k):
        self.weight = limit_weight(self.__weight, k)
        return self

    def eval(self):
        self.__weight = self.weight

    def reset(self):
        self.weight = self.__weight
