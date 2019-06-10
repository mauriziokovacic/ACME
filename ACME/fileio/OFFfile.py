import torch
import warnings
from ACME.utility.row           import *
from ACME.utility.col           import *
from ACME.utility.numel         import *
from ACME.utility.debug_message import *
from ACME.utility.tensor2string import *
from ACME.utility.string2tensor import *
from ACME.utility.transpose     import *
from ACME.topology.poly2edge    import *
from .io                        import *



def export_OFF(filename,Point=None,Face=None,Edge=None,computeEdge=False,verbose=False):
    """
    Exports the given mesh data into an OFF file

    Parameters
    ----------
    filename : str
        name of the file
    Point : Tensor (optional)
        the mesh vertex positions (default is None)
    Face : LongTensor (optional)
        the topology tensor (default is None)
    Edge : LongTensor (optional)
        the edge tensor (default is None)
    computeEdge : bool (optional)
        if True and edges were not provided, computes the edges of the input mesh
    verbose : bool
        if True executes the function writing debug messages to the console

    Returns
    -------
    None
    """

    if computeEdge and (Edge is not None):
        Edge = poly2edge(F)[0]
    nP = 0 if Point is None else row(Point)
    nF = 0 if Face  is None else col(Face)
    nE = 0 if Edge  is None else col(Edge)
    def writeDataFcn(fileID):
        content = ''
        debug_message('Creating the header...',verbose,no_newline=True)
        content = 'OFF\n{} {} {}\n'.format(nP,nF,nE)
        debug_message('DONE.',verbose)
        if Point is not None:
            debug_message('Creating vertices...',verbose,no_newline=True)
            content += tensor2string(Point)+'\n'
            debug_message('DONE.',verbose)
        if Face is not None:
            debug_message('Creating faces...',verbose,no_newline=True)
            content += tensor2string(torch.t(Face.cpu()),prefix='{} '.format(row(Face)))+'\n'
            debug_message(['DONE.',newline],verbose)
        if Edge is not None:
            debug_message('Creating edges...',verbose,no_newline=True)
            content += tensor2string(torch.t(Edge))+'\n'
            debug_message(['DONE.',newline],verbose)
        fileID.write(content)
        debug_message('Data writing completed.',verbose)
        return
    export_to_text_file(filename,'off',writeDataFcn,verbose=verbose)
    return



def import_OFF(filename,device='cuda:0',verbose=False):
    """
    Imports the data from the given OFF file

    Parameters
    ----------
    filename : str
        name of the file
    device : str or torch.device (optional)
        the device the data will be stored to (default is 'cuda:0')
    verbose : bool
        if True executes the function writing debug messages to the console

    Returns
    -------
    (Tensor,LongTensor,LongTensor)
        the mesh vertices, the topology tensor and the edge tensor
    """

    def readDataFcn(fileID):
        content = fileID.read()
        # tokenize
        content = content.split('\n')
        # delete empty lines
        content = list(x for x in content if len(x)!=0)
        # delete comments
        content = list(x for x in content if x[0]!='#')
        # check header
        if content[0]!='OFF':
            warnings.warn('File {} is not a valid OFF.'.format(fileID.name()),RuntimeWarning)
            return
        # read data size
        v,f,e = [int(x) for x in content[1].split()]
        # read data
        V = string2tensor('\n'.join(content[2:2+v]),dtype=torch.float,device=device)
        F = string2tensor('\n'.join(content[2+v:2+v+f]),dtype=torch.long,device=device)
        F = torch.t(F[:,1:]) if numel(F)>0 else F
        E = string2tensor('\n'.join(content[2+v+f:2+v+f+e]),dtype=torch.long,device=device)
        return V,F,E
    return import_from_text_file(filename,'off',readDataFcn,verbose=verbose)
