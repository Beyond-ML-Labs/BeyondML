import torch


class MultiConv2D(torch.nn.Module):
    """
    Multi- 2D Convolutional layer initialized with weights rather than hyperparameters
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
            The bias matrix to use
        padding : str or int (default 'same')
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
        outputs = []
        kernel = self.w
        bias = self.b

        for i in range(len(inputs)):
            outputs.append(
                torch.nn.functional.conv2d(
                    inputs[i],
                    kernel[i],
                    bias[i],
                    stride=self.strides,
                    padding=self.padding
                )
            )
        return outputs
