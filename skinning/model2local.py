import torch
from ..utility.isempty  import *
from ..utility.numel    import *
from ..topology.digraph import *


def model2local(graph, model_pose):
    """
    Returns the local space pose from the given hierarchy and model space pose

    Parameters
    ----------
    graph : Digraph
        a directed graph
    model_pose : Tensor
        a (N,K,K,) tensor representing the model space pose

    Returns
    -------
    Tensor
        a (N,K,K,) tensor representing the local space pose

    Raises
    ------
    AssertionError
        if graph is not a Digraph
    """

    assert isinstance(graph, Digraph), 'graph should be a Digraph. Got {} instead.'.format(type(graph))
    local_pose = torch.zeros_like(model_pose)
    root = graph.roots()
    for r in root:
        n = [r]
        local_pose[n, :, :] = model_pose[n, :, :]
        while len(n) > 0:
            i = n[0]
            n = n[1:]
            child = graph.successors(i)
            for j in child:
                local_pose[j, :, :] = torch.matmul(model_pose[i, :, :], torch.inverse(model_pose[j, :, :]))
                n += [j]
    return local_pose
