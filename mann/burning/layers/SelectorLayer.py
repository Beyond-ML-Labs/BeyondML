from torch.nn import Module


class SelectorLayer(Module):
    """
    Layer which selects an individual input based on index and only returns that one
    """

    def __init__(
        self,
        sel_index
    ):
        """
        Parameters
        ----------
        sel_index : int
            The index of inputs to select
        """
        super().__init__()
        self.sel_index = sel_index

    @property
    def sel_index(self):
        return self._sel_index

    @sel_index.setter
    def sel_index(self, value):
        if not isinstance(value, int):
            raise TypeError('sel_index must be integer-valued')
        self._sel_index = value

    def forward(self, inputs):
        """
        Call the layer on input data

        Parameters
        ----------
        inputs : torch.Tensor
            Inputs to call the layer's logic on

        Returns
        -------
        results : torch.Tensor
            The results of the layer's logic
        """
        return inputs[self.sel_index]
