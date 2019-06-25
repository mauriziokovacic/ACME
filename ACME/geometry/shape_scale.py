from ..math.norm   import *
from .bounding_box import *

def shape_scale(P,dim=0):
    """
    Returns the diagonal of the input point set bounding box along the specified dimension

    Parameters
    ----------
    P : Tensor
        the input point set
    dim : int (optional)
        the dimension along the scale is computed

    Returns
    -------
    float
        the length of the point set bounding box diagonal
    """

    min,max = bounding_box(P,dim=dim)
    return norm(max-min)
