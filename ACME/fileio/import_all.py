import os
import warnings


def import_all(path, error=False):
    """
    Imports all modules in the given path

    Parameters
    ----------
    path : str
        the path where to import the modules from
    error : bool (optional)
        if True, whenever an import cannot be made, an exception will be raised

    Returns
    -------
    None

    Raises
    ------
    ImportError
        if error is set to True and a module cannot be imported
    """

    for module in os.listdir(path):
        if not module.startswith('__') and module.endswith('.py'):
            module = module[:-3]
            if error:
                exec('from .{} import *'.format(module))
            else:
                try:
                    exec('from .{} import *'.format(module))
                except ImportError:
                    warnings.warn('Module {} could not be imported'.format(module))
