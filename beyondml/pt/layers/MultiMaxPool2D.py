import torch


class MultiMaxPool2D(torch.nn.Module):
    """
    Multitask implementation of 2-dimensional Max Pooling layer
    """

    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        dilation=1
    ):
        """
        Parameters
        ----------
        kernel_size : int or tuple
            The kernel size to use
        stride : int, tuple, or None (default None)
            The stride to use.  If None, defaults to kernel_size
        padding : int (default 0)
            The padding to use
        dilation : int (default 1)
            The dilation to use
        """

        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride else self.kernel_size
        self.padding = padding
        self.dilation = dilation

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
            outputs.append(
                torch.nn.functional.max_pool2d(
                    input=inputs[i],
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation
                )
            )
        return outputs
