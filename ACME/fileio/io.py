from ACME.utility.strcmpi       import *
from ACME.utility.debug_message import *
from .fileparts                 import *


def _open_file(filename,defaultExt,permission,fileDataFcn,verbose=False):
    """
    Opens a given file and executes the specified function over its data

    Parameters
    ----------
    filename : str
        the path of a file
    defaultExt : str
        the default extension of the file
    permission : str
        a string representing how to open the file (Ex.: 'r','w',...)
    fileDataFcn : callable
        the function to be processed over file data. Takes the file as its only parameter.
    verbose : bool (optional)
        if True prints messages on console (default is False)

    Returns
    -------
    unknown
        the output of the fileDataFcn
    """

    path,filename,ext = fileparts(filename)
    if (not ext) or (not strcmpi(ext[1:-1],defaultExt)):
        ext = '.'+defaultExt
    debug_message('Opening file "{}/{}{}"...'.format(path,filename,ext),verbose,no_newline=True)
    try:
        fileID = open(path+'/'+filename+ext,mode=permission)
    except:
        debug_message('FAILED.',verbose)
        ext = '.'+defaultExt.upper()
        debug_message('Opening file "{}/{}{}"...'.format(path,filename,ext),verbose,no_newline=True)
        try:
            fileID = open(path+'/'+filename+ext,mode=permission)
        except:
            debug_message('FAILED.',verbose)
            warn('File "{}/{}{}"  does not exist.')
            return
    debug_message('DONE.',verbose)
    debug_message('Starting routine...',verbose)
    out = fileDataFcn(fileID)
    debug_message('Routine COMPLETED.',verbose)
    debug_message('Closing file...',verbose,no_newline=True)
    fileID.close()
    debug_message('COMPLETED.',verbose)
    return out



def export_to_text_file(filename,defaultExt,writeDataFcn,verbose=False):
    """
    Opens a given text file and writes data using the specified function

    Parameters
    ----------
    filename : str
        the path of a file
    defaultExt : str
        the default extension of the file
    writeDataFcn : callable
        the function to write data to the file. Takes the file as its only parameter.
    verbose : bool (optional)
        if True prints messages on console (default is False)

    Returns
    -------
    unknown
        the output of the writeDataFcn
    """

    return _open_file(filename,defaultExt,'w+',writeDataFcn,verbose=verbose)



def export_to_binary_file(filename,defaultExt,writeDataFcn,verbose=False):
    """
    Opens a given binary file and writes data using the specified function

    Parameters
    ----------
    filename : str
        the path of a file
    defaultExt : str
        the default extension of the file
    writeDataFcn : callable
        the function to write data to the file. Takes the file as its only parameter.
    verbose : bool (optional)
        if True prints messages on console (default is False)

    Returns
    -------
    unknown
        the output of the writeDataFcn
    """

    return _open_file(filename,defaultExt,'wb',writeDataFcn,verbose=verbose)



def import_from_text_file(filename,defaultExt,readDataFcn,verbose=False):
    """
    Opens a given text file and reads data using the specified function

    Parameters
    ----------
    filename : str
        the path of a file
    defaultExt : str
        the default extension of the file
    readDataFcn : callable
        the function to read data from the file. Takes the file as its only parameter.
    verbose : bool (optional)
        if True prints messages on console (default is False)

    Returns
    -------
    unknown
        the output of the readDataFcn
    """

    return _open_file(filename,defaultExt,'r',readDataFcn,verbose)



def import_from_binary_file(filename,defaultExt,readDataFcn,verbose=False):
    """
    Opens a given text file and reads data using the specified function

    Parameters
    ----------
    filename : str
        the path of a file
    defaultExt : str
        the default extension of the file
    readDataFcn : callable
        the function to be read data from the file. Takes the file as its only parameter.
    verbose : bool (optional)
        if True prints messages on console (default is False)

    Returns
    -------
    unknown
        the output of the readDataFcn
    """

    return _open_file(filename,defaultExt,'rb',readDataFcn,verbose)



def append_to_text_file(filename,defaultExt,writeDataFcn,verbose=False):
    """
    Opens a given text file and appends data using the specified function

    Parameters
    ----------
    filename : str
        the path of a file
    defaultExt : str
        the default extension of the file
    writeDataFcn : callable
        the function to write data to the file. Takes the file as its only parameter.
    verbose : bool (optional)
        if True prints messages on console (default is False)

    Returns
    -------
    unknown
        the output of the writeDataFcn
    """

    return _open_file(filename,defaultExt,'a+',writeDataFcn,verbose)



def append_to_binary_file(filename,defaultExt,writeDataFcn,verbose=False):
    """
    Opens a given binary file and appends data using the specified function

    Parameters
    ----------
    filename : str
        the path of a file
    defaultExt : str
        the default extension of the file
    writeDataFcn : callable
        the function to write data to the file. Takes the file as its only parameter.
    verbose : bool (optional)
        if True prints messages on console (default is False)

    Returns
    -------
    unknown
        the output of the writeDataFcn
    """

    return _open_file(filename,defaultExt,'ab',writeDataFcn,verbose)
