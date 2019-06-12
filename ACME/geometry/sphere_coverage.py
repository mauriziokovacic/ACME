import torch
from ACME.utility.indices import *
from ACME.math.constant   import *
from ACME.math.cos        import *
from ACME.math.sin        import *



def sphere_coverage(n,dtype=torch.float,device='cuda:0'):
    """
    Returns a almost-uniform sampling of a sphere, using the Fibonacci formula

    Parameters
    ----------
    n : int
        the number of samples on the sphere surface
    dtype : type (optional)
        the type of the output tensor (default is torch.float)
    device : str or torch.device (optional)
        the device the tensor will be stored to (default is 'cuda:0')

    Returns
    -------
    Tensor
        the points set tensor
    """

    i    = indices(0,n-1,dtype=dtype,device=device)
    phi  = i/SQRT3
    phi  = (phi-torch.floor(phi))*PI2
    cosT = (i*2+1)/n-1
    sinT = torch.sqrt(torch.clamp(1-torch.pow(cosT,2),min=0,max=1))
    return torch.cat((cos(phi)*sinT,sin(phi)*sinT,cosT),dim=1)




