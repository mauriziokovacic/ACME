import torch
from ..utility.isempty  import *
from ..utility.numel    import *
from ..topology.digraph import *


def local2model(graph, local_pose):
    """
    Returns the model space pose from the given hierarchy and local space pose

    Parameters
    ----------
    graph : Digraph
        a directed graph
    local_pose : Tensor
        a (N,K,K,) tensor representing the local space pose

    Returns
    -------
    Tensor
        a (N,K,K,) tensor representing the model space pose
    """

    model_pose = torch.zeros_like(local_pose)
    root = graph.roots()
    for r in root:
        n = [r]
        model_pose[n, :, :] = local_pose[n, :, :]
        while len(n) > 0:
            i = n[0]
            n = n[1:]
            child = graph.successors(i)
            for j in child:
                model_pose[j, :, :] = model_pose[i, :, :] * local_pose[j, :, :]
                n += [j]
    return model_pose
