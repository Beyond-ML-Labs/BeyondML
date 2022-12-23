import torch


class FilterLayer(torch.nn.Module):
    """
    Layer which filters input data, either returning values or all zeros depending on state
    """

    def __init__(
        self,
        is_on=True,
        device=None,
        dtype=None
    ):
        """
        Parameters
        ----------
        is_on : bool (default False)
            Whether the layer is on or off
        """

        super().__init__()
        self.is_on = is_on
        self.factory_kwargs = {'device': device, 'dtype': dtype}

    @property
    def is_on(self):
        return self._is_on

    @is_on.setter
    def is_on(self, value):
        if not isinstance(value, bool):
            raise TypeError('is_on must be Boolean')
        self._is_on = value

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
        if self.is_on:
            return inputs
        else:
            return torch.zeros_like(inputs, **self.factory_kwargs)

    def turn_on(self):
        """
        Turn on the layer
        """
        self.is_on = True

    def turn_off(self):
        """
        Turn off the layer
        """
        self.is_on = False
