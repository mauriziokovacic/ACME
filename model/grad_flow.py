def grad_flow(model):
    """
    Returns a dictionary containing the [min, mean, max] of each layer gradient

    Parameters
    ----------
    model : torch.nn.Module
        the module to extract the gradient flow from

    Returns
    -------
    dict
        a dictionary with the key being the layer name and its value being a [min, mean, max] list
    """

    d = {}
    for n, p in model.named_parameters():
        if p.requires_grad and ("bias" not in n):
            g = p.grad.abs()
            d[n] = [g.min().item(), g.mean().item(), g.max().item()]
    return d