import torch
from .flatten import *

def setdiff(A,B)
    return torch.tensor(list(set(flatten(A))-set(flatten(B))),dtype=A.dtype,device=A.device)
