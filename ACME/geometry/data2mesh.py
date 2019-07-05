def data2mesh(data):
    """
    Extracts from a given torch_geometric Data object the mesh elements

    Parameters
    ----------
    data : Data
        a torch_geometric Data object

    Returns
    -------
    (Tensor,LongTensor,Tensor)
        the points set, the topology and the vertex normals tensor
    """

    return data.pos, data.face, data.norm
