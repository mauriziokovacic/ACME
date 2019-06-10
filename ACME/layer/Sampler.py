import torch
from ACME.color.fetch_texture import *


class Sampler1D(torch.nn.Module):
    def __init__(self):
        super(Sampler,self).__init__()

    def forward(self,data,param):
        return fetch_texture1D(data,param,mode='bilinear')



class Sampler2D(torch.nn.Module):
    def __init__(self):
        super(Sampler,self).__init__()

    def forward(self,data,param):
        return fetch_texture1D(data,param,mode='bilinear')



class Sampler3D(torch.nn.Module):
    def __init__(self):
        super(Sampler,self).__init__()

    def forward(self,data,param):
        return fetch_texture1D(data,param,mode='bilinear')
