import torch
from ACME.utility.row           import *
from ACME.utility.tensor2string import *
from ACME.color.color2int       import *
from .io                        import *

def export_PLY(filename,Point=None,Normal=None,UV=None,Face=None,Color=None,verbose=False):
    """
    Exports the given mesh data into an OBJ file

    Parameters
    ----------
    filename : str
        name of the file
    Point : Tensor (optional)
        the mesh vertex positions (default is None)
    Normal : Tensor (optional)
        the mesh vertex normals (default is None)
    UV : Tensor (optional)
        the mesh vertex UVs (default is None)
    Face : LongTensor (optional)
        the topology tensor (default is None)
    Color : Tensor or IntTensor (optional)
        the mesh vertex colors (default is None)
    verbose : bool
        if True executes the function writing debug messages to the console

    Returns
    -------
    None
    """

    header  = 'ply\nformat ascii 1.0\n'
    header += 'element vertex {}\n'.format(row(Point) if Point is not None else 0)
    if Point is not None:
        header += 'property float x\n'
        header += 'property float y\n'
        header += 'property float z\n'
        if Normal is not None:
            header += 'property float nx\n'
            header += 'property float ny\n'
            header += 'property float nz\n'
        if UV is not None:
            header += 'property float u\n'
            header += 'property float v\n'
        if Color is not None:
            header += 'property uchar red\n'
            header += 'property uchar green\n'
            header += 'property uchar blue\n'
    header += 'element face {}\n'.format(row(Face) if Face is not None else 0)
    if Face is not None:
        header += 'property list uchar int vertex_index\n'
    header += 'end_header\n'
    def writeDataFcn(fileID):
        content = header
        if Point is not None:
            content += tensor2string(Point)
            content += tensor2string(Normal)
            content += tensor2string(UV)
            content += tensor2string(color2int(Color) if Color is not None else Color)
        if Face is not None:
            content += tensor2string(torch.t(Face), prefix='{} '.format(row(Face)))
        fileID.write(content)
        return
    export_to_text_file(filename,'ply',writeDataFcn,verbose=verbose)
    return
