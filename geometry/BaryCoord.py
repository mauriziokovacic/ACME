from ..utility.find import *


class BaryCoord(torch.Tensor):
    def __init__(self, *args, **kwargs):
        super(BaryCoord, self).__init__()
        self.to(dtype=torch.float)

    def is_valid(self):
        return (torch.sum(self, -1, keepdim=True) == 1).squeeze()

    def is_invalid(self):
        return 1 - self.is_valid()

    def is_outside(self):
        return (torch.sum((self < 0) + (self > 1), -1, keepdim=True) > 0).squeeze()

    def is_inside(self):
        return 1 - self.is_outside()

    def is_on_face(self):
        return self.is_inside()

    def is_on_vertex(self):
        tf = self.is_on_face()
        bc = self == 1
        tf = tf * (torch.sum(bc, -1, keepdim=True) == 1).squeeze()
        if self.ndimension() == 1:
            bc = bc.unsqueeze(1)
            I = -torch.ones(1, dtype=torch.long, device=self.device)
        else:
            I = -torch.ones(self.shape[0], dtype=torch.long, device=self.device)
        i = find(bc[tf], linear=False).t().squeeze()
        I[i[0]] = i[1]
        return tf, I

    def is_on_edge(self):
        tf = self.is_on_face()
        bc = self == 0
        tf = tf * (torch.sum(bc, -1, keepdim=True) == 1).squeeze()
        if self.ndimension() == 1:
            bc = bc.unsqueeze(1)
            I  = -torch.ones(1, dtype=torch.long, device=self.device)
        else:
            I = -torch.ones(self.shape[0], dtype=torch.long, device=self.device)
        i = find(bc[tf], linear=False).t().squeeze()
        I[i[0]] = (i[1] + 1) % 3
        return tf, I

    def eval(self, P, T):
        return torch.sum(P[T] * (self if self.ndimension()>1 else self.unsqueeze(0)).t().unsqueeze(P.ndimension()), dim=0)
