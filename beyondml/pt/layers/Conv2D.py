import torch


class Conv2D(torch.nn.Module):
    """
    Convolutional 2D layer initialized directly with weights, rather than with hyperparameters
    """

    def __init__(
        self,
        kernel,
        bias,
        padding='same',
        strides=1,
        device=None,
        dtype=None
    ):
        """
        Parameters
        ----------
        kernel : torch.Tensor or Tensor-like
            The kernel tensor to use
        bias : torch.Tensor or Tensor-like
            The bias tensor to use
        padding : int or str (default 'same')
            The padding to use
        strides : int or tuple (default 1)
            The strides to use
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.w = torch.nn.Parameter(torch.Tensor(kernel).to(**factory_kwargs))
        self.b = torch.nn.Parameter(torch.Tensor(bias).to(**factory_kwargs))

        self.padding = padding
        self.strides = strides

    def forward(
        self,
        inputs
    ):
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

        return torch.nn.functional.conv2d(
            inputs,
            self.w,
            self.b,
            stride=self.strides,
            padding=self.padding
        )
