import torch
from utility.indices import *
from math.constant   import *
from math.cos        import *
from math.sin        import *


def sphere_coverage(nsamples,dtype=torch.float,device='cuda:0'):
    i    = indices(0,nsamples-1,dtype=dtype,device=device)
    phi  = i/SQRT3
    phi  = (phi-torch.floor(phi))*PI2
    cosT = (i*2+1)/nsamples-1
    sinT = torch.sqrt(torch.clamp(1-torch.pow(c,2),min=0,max=1))
    return torch.cat((cos(phi)*sinT,sin(phi)*sinT,cosT),dim=1)




