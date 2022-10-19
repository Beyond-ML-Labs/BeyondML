import torch


class SparseMultiConv3D(torch.nn.Module):
    """
    Sparse implementation of a Multitask 3D Convolutional layer, expected to be converted from a
    trained, pruned layer
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
        strides : int or tuple (default 1)
            The strides to use
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

        kernel = self.w.to_dense()
        bias = self.b.to_dense()

        outputs = []
        for i in range(len(inputs)):
            outputs.append(
                torch.nn.functional.conv3d(
                    inputs[i],
                    kernel[i],
                    bias[i],
                    stride=self.strides,
                    padding=self.padding
                )
            )
        return outputs
