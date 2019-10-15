def plot_mesh(session, P, T, win='Mesh', opts=None, **kwargs):
    """
    Plots the given mesh on visdom

    Parameters
    ----------
    session : Visdom
        the visdom session
    P : Tensor
        a (N,3,) points set tensor
    T : LongTensor
        a (3,M,) topology tensor
    win : str (optional)
        the name of the visdom window (default is 'Mesh')
    opts : dict (optional)
        if given, adds the specified options (default is None)
    kwargs : ...

    Returns
    -------
    object
        the plot object
    """

    return session.mesh(P, T, win=win, opts={'title': win}.update(opts if opts is not None else dict()), **kwargs)
