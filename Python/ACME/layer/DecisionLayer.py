import torch



class DecisionLayer(torch.nn.Module):
    """
    A fully connected layer performing a decision

    Attributes
    ----------
    model : torch.nn.Sequential
        the decision architecture

    Methods
    -------
    __create_model(input_size,output_size,bottleneck)
        creates the decision architecture
    forward(input)
        returns the decision tensor
    """

    def __init__(self,input_size,output_size,bottleneck=4096):
        """
        Parameters
        ----------
        input_size : int
            the input size
        output_size : int
            the output size
        bottleneck : int (optional)
            the size of the initial layer output (default is 4096)
        """

        super(DecisionLayer,self).__init__()
        self.__create_model(input_size,output_size,bottleneck)



    def __create_model(self,input_size,output_size,bottleneck):
        """
        Creates the decision architecture

        Parameters
        ----------
        input_size : int
            the input size
        output_size : int
            the output size
        bottleneck : int
            the size of the initial layer output

        Returns
        -------
        None
        """

        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, bottleneck),
            torch.nn.ReLU(True),
            torch.nn.Dropout(),
            torch.nn.Linear(bottleneck,bottleneck//2),
            torch.nn.ReLU(True),
            torch.nn.Dropout(),
            torch.nn.Linear(bottleneck//2,output_size),
        )



    def forward(self,input):
        """
        Returns the decision tensor

        Parameters
        ----------
        input : Tensor
            the input tensor

        Returns
        -------
        Tensor
            the decision tensor
        """

        return self.model(inputs)



class ClassifierLayer(DecisionLayer):
    """
    A fully connected layer performing a classification
    """

    def __init__(self,input_size,n_classes,bottleneck=4096):
        """
        Parameters
        ----------
        input_size : int
            the input size
        n_classes : int
            the number of classes in the classification
        bottleneck : int (optional)
            the size of the initial layer output (default is 4096)
        """

        super(ClassifierLayer,self).__init__(input_size,n_classes,bottleneck=bottleneck)
        self.model = torch.nn.Sequential(
            self.model,
            torch.nn.Softmax(),
        )
