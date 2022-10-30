import torch


class Dense(torch.nn.Module):
    """
    Fully-connected layer initialized directly with weights, rather than hyperparameters
    """

    def __init__(
        self,
        weight,
        bias,
        device=None,
        dtype=None
    ):
        """
        Parameters
        ----------
        weight : torch.Tensor or Tensor-like
            The weight matrix to use
        bias : torch.Tensor or Tensor-like
            The bias vector to use
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.w = torch.nn.Parameter(torch.Tensor(weight).to(**factory_kwargs))
        self.b = torch.nn.Parameter(torch.Tensor(bias).to(**factory_kwargs))

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
        out = torch.mm(inputs, self.w)
        out = torch.add(out, self.b)
        return out
