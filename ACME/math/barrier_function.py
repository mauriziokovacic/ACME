from .constant import *

def barrier_function(x,t):
    """
    Returns the barrier function for a given value x and a threshold t

    Parameters
    ----------
    x : Tensor
        the input value
    t : scalar
        the threshold value from which the barrier starts

    Returns
    -------
    Tensor
        the barrier value
    """

    def g(x):
        return (x**3)/(t**3) - 3*(x**2)/(t**2) + 3*x/t

    out       = torch.zeros_like(x,dtype=torch.float,device=x.device)
    i         = (x>0)and(x<t)
    out[i]    = torch.reciprocal(g(x[i])) - 1
    out[x<=0] = Inf
    return out
