import torch
from ..utility.Uint8Tensor import *
from .color2float          import *

def black():
    return torch.zeros(1,3)

def white():
    return torch.ones(1,3)

def red():
    return color2float(Uint8Tensor([237,28,36]))

def green():
    return color2float(Uint8Tensor([34,177,76]))

def blue():
    return color2float(Uint8Tensor([0,162,232]))

def cyan():
    return torch.add(torch.neg(red()),1)

def magenta():
    return torch.add(torch.neg(green()),1)

def yellow():
    return torch.add(torch.neg(blue()),1)

def brown():
    return color2float(Uint8Tensor([149,116,83]))

def darkteal():
    return color2float(Uint8Tensor([98,140,178]))

def grey():
    return torch.mul(torch.ones(1,3),0.5)

def orange():
    return color2float(Uint8Tensor([253,135,86]))

def pink():
    return color2float(Uint8Tensor([254,194,194]))

def teal():
    return color2float(Uint8Tensor([144,216,196]))
