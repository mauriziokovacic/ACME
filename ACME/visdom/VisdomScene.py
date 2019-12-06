from .create_session import *


class VisdomScene(object):
    """
    A class defining a visdom environment scene

    Attributes
    ----------
    __session : Visdom
        the visdom session
    __plot : dict
        the scene plots

    Methods
    -------
    size()
        returns the number of plots in the scene
    is_empty()
        returns True if the scene is empty, False otherwise
    insert_plot(name, cls, *args, **kwargs)
        inserts a plot specifying class and input parameters
    remove_plot(name)
        removes the plot specified by its name
    update_plot(name, *args, **kwargs)
        updates the plot specified by its name, passing its given arguments
    """

    def __init__(self, env='main'):
        """
        Parameters
        ----------
        env : str (optional)
            the environment name (default is 'main')
        """

        self.__session = create_session(env=env)
        self.__plot  = {}

    def size(self):
        """
        Returns the number of plots in the scene

        Returns
        -------
        int
            the number of plots
        """

        return len(self.__plot)

    def is_empty(self):
        """
        Returns True if the scene is empty, False otherwise

        Returns
        -------
        bool
            True if the scene is empty, False otherwise
        """

        return self.size() == 0

    def insert_plot(self, name, cls, *args, **kwargs):
        """
        inserts a plot specifying class and input parameters

        Parameters
        ----------
        name : str
            a unique name for the plot to add
        cls : class
            the class of the plot to add
        args : ...
            the argument for the plot constructor
        kwargs : ...
            the keyword argument for the plot constructor

        Returns
        -------
        VisdomScene
            the scene itself
        """

        self.__plot[name] = cls(self.__session, *args, win=name, **kwargs)
        return self

    def remove_plot(self, name):
        """
        Removes the plot specified by its name

        Parameters
        ----------
        name : str
            the unique name of the plot

        Returns
        -------
        VisdomScene
            the scene itself
        """

        if name in self.__plot:
            self.__plot[name].close()
            del self.__plot[name]
        return self

    def update_plot(self, name, *args, **kwargs):
        """
        Updates the plot specified by its name, passing its given arguments

        Parameters
        ----------
        name : str
            the unique name of the plot
        args : ...
            the arguments for the plot update method
        kwargs : ...
            the keyword arguments for the plot update method

        Returns
        -------
        VisdomScene
            the scene itself
        """

        self.__plot[name].update(*args, **kwargs)
        return self
