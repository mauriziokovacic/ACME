def plot_image(session, images, win='Image', **kwargs):
    """
    Plots the given images in visdom

    Parameters
    ----------
    session : Visdom
        the visdom session
    images : Tensor
        the images tensor
    win : str (optional)
        the window name (default is 'Image')
    kwargs : ...

    Returns
    -------
    object
        the visdom plot
    """

    return session.images(images, win=win, **kwargs)
