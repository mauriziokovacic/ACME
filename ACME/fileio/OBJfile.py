import torch
from ACME.utility.row           import *
from ACME.utility.repelem       import *
from ACME.utility.debug_message import *
from ACME.utility.tensor2string import *
from ACME.utility.string2tensor import *
from .io                        import *


def export_OBJ(filename,Point=None,Normal=None,UV=None,Face=None,verbose=False):
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
    verbose : bool
        if True executes the function writing debug messages to the console

    Returns
    -------
    None
    """

    def face_format(uv,normal):
        s = '{}';
        if normal:
            s = s+'/';
            if(uv):
                s = s,'{}';
            s = s+'/{}';
        return s
    def writeDataFcn(fileID):
        content = ''
        if Point is not None:
            content += tensor2string(Point,prefix='v ') + '\n'
        hasN  = Normal is not None
        hasUV = UV is not None
        if hasN:
            content += tensor2string(Normal,prefix='vn ') + '\n'
        if hasUV:
            content += tensor2string(UV,prefix='vt ') + '\n'
        if Face is not None:
            str = (face_format(hasUV,hasN)+' ')*row(Face)
            str = 'f ' + str[0:-1] + '\n'
            for f in repelem(torch.t(Face).cpu(),1,1+hasUV+hasN):
                content += (str.format(*tuple(f.numpy()+1)))
        fileID.write(content)
        return
    export_to_text_file(filename,'obj',writeDataFcn,verbose=verbose)
    return



def import_OBJ(filename,device='cuda:0',verbose=False):
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
    (Tensor,Tensor,Tensor,LongTensor)
        the mesh vertices, normals,uv and the topology tensor
    """

    def readDataFcn(fileID):
        # tokenize
        content = txt.split('\n')
        # delete empty liens
        content = list(x for x in content if len(x)!=0)
        # delete comments
        content = list(x for x in content if x[0]!='#')
        # read vertex data
        V  = string2tensor('\n'.join(list(x for x in content if x[0:2]=='v ' )), prefix='v ',dtype=torch.float,device=device)
        # read uv data
        UV = string2tensor('\n'.join(list(x for x in content if x[0:3]=='vt ')), prefix='vt ',dtype=torch.float,device=device)
        # read normal data
        N  = string2tensor('\n'.join(list(x for x in content if x[0:3]=='vn ')), prefix='vn ',dtype=torch.float,device=device)
        # isolate face data
        content = list(x for x in content if x[0]=='f')
        content = list(x.replace('f ','').split() for x in content)
        content = '\n'.join(list(' '.join(list(i.split('/')[0] for i in x )) for x in content))
        # read face data
        F = torch.t(string2tensor(content,dtype=torch.long,device=device))-1
        return V,N,UV,F
    return import_from_text_file(filename,'obj',readDataFcn,verbose=verbose)
