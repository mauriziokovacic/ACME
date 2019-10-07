import torch


class TLayer(torch.nn.Module):
    def __init__(self, K=3, bias=False):
        super(TLayer, self).__init__()
        self.weight = torch.nn.Parameter(data=torch.eye(K, dtype=torch.float, requires_grad=True), requires_grad=True)
        if bias:
            self.bias = torch.nn.Parameter(data=torch.eye(K, dtype=torch.float, requires_grad=True), requires_grad=True)
        else:
            self.register_parameter('bias', None)
        self.init_parameters()

    def init_parameters(self):
        if self.bias is not None:
            self.weight.data.fill_(0)

    def forward(self, x):
        return torch.matmul(x, self.weight) + self.bias
