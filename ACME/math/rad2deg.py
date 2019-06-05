def rad2deg(radians):
    """
    Computes the degree value from input radians

    Parameters
    ----------
    radians : int,float or Tensor
        input radians values

    Returns
    -------
    float or Tensor
        the degree values
    """

    return (radians*180)/PI
