import torch


class MultitaskNormalization(torch.nn.Module):
    """
    Layer which normalizes a set of inputs to sum to 1
    """

    def __init__(
        self,
        device=None,
        dtype=None
    ):

        super().__init__()
        self.factory_kwargs = {'device': device, 'dtype': dtype}

    def forward(self, inputs):
        """
        Call the layer on input data

        Parameters
        ----------
        inputs : torch.Tensor or list of Tensors
            Inputs to call the layer's logic on

        Returns
        -------
        results : torch.Tensor or list of Tensors
            The results of the layer's logic
        """
        s = 0
        for i in inputs:
            s += 1
        return [i / s for i in inputs]
