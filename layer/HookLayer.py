import torch
from ..model.hook import *


class HookLayer(torch.nn.Module):
    def __init__(self, layer=None):
        super(HookLayer, self).__init__()
        self.__hook = Hook(layer=layer, outputFcn=self.__outputFcn)

    def is_bound(self):
        return self.__hook.is_bound()

    def bind(self, layer):
        self.__hook.bind(layer)

    def unbind(self):
        self.__hook.unbind()

    def __outputFcn(self, output):
        self.__output = output

    def forward(self, x, y):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, self.__output, **kwargs)