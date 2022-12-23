import torch


class MultiDense(torch.nn.Module):
    """
    Multi-Fully-Connected layer initialized with weights rather than hyperparameters
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
            The weight tensor to use
        bias : torch.Tensor or Tensor-like
            The bias tensor to use
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
        outputs = []
        for i in range(len(inputs)):
            out = torch.mm(inputs[i], self.w[i])
            out = torch.add(out, self.b[i])
            outputs.append(out)
        return outputs
