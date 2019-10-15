import numpy


def plot_grad(session, grad, win='Grad', **kwargs):
    """
    Plots the grad norms into visdom

    Parameters
    ----------
    session : Visdom
        a visdom session
    grad : dict
        the grad dict coming from model.grad_flow
    win : str (optional)
        the window name (default is 'Grad')
    kwargs : ...

    Returns
    -------
    object
        the plot object
    """

    x = []
    y = []
    for i, value in enumerate(grad.values()):
        x += [value[0]]
        y += [i]
    session.line(X=numpy.array(x),
                 Y=numpy.array(y),
                 win=win,
                 opts=dict(
                     title=win,
                     xlabel='Layers',
                     ylabel='Norm', ),
                 **kwargs,
                 )
