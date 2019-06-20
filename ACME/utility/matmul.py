import numpy
import torch
from .isnumpy  import *
from .istorch  import *
from .issparse import *

def matmul(tensor_a,tensor_b):
    if isnumpy(tensor_a,tensor_b):
        return numpy.matmul(tensor_a,tensor_b)
    if istorch(tensor_a,tensor_b):
        if issparse(tensor_a):
            return torch.sparse.mm(tensor_a,tensor_b)
        return torch.mm(tensor_a,tensor_b)
    assert False, 'Unknown data type'
