from visdom import Visdom


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

    session = Visdom(**kwargs)
    session.close(None)
    return session
