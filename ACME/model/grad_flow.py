def grad_flow(model):
    """
    Returns a dictionary containing the mean of each layer gradient

    Parameters
    ----------
    model : torch.nn.Module
        the module to extract the gradient flow from

    Returns
    -------
    dict
        a dictionary with the key being the layer name and its value being the layer gradient mean
    """

    d = {}
    for n, p in model.named_parameters():
        if p.requires_grad and ("bias" not in n):
            g = p.grad
            if g is not None:
                d[n] = g.abs().mean().item()
            else:
                d[n] = 0
    return d
