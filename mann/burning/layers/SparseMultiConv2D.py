import torch


class SparseMultiConv2D(torch.nn.Module):
    """
    Sparse implementation of a Multi 2D Convolutional layer
    """

    def __init__(
            self,
            kernel,
            bias,
            padding='same',
            strides=1
    ):
        """
        Parameters
        ----------
        kernel : torch.Tensor or Tensor-like
            The kernel to use
        bias : torch.Tensor or Tensor-like
            The bias to use
        padding : str or int (default 'same')
            The padding to use
        strides : int (default 1)
            The padding to use
        """
        super().__init__()
        self.w = torch.Tensor(kernel).to_sparse()
        self.b = torch.Tensor(bias).to_sparse()
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
        kernel = self.w.to_dense()
        bias = self.b.to_dense()

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
