import visdom


def create_visdom_session(**kwargs):
    """
    Creates a visdom session

    Parameters
    ----------
    kwargs : ...

    Returns
    -------
    object
        a visdom session
    """

    session = visdom.Visdom(**kwargs)
    session.close(None)
    return session
