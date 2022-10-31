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
        strides=1,
        device=None,
        dtype=None
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

        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        for i in range(kernel.shape[0]):
            self.register_buffer(
                f'w_{i}',
                torch.Tensor(kernel[i]).to(**factory_kwargs).to_sparse()
            )
            self.register_buffer(
                f'b_{i}',
                torch.Tensor(bias[i]).to(**factory_kwargs).to_sparse()
            )

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
        for i in range(len(inputs)):
            outputs.append(
                torch.nn.functional.conv3d(
                    inputs[i],
                    self.get_buffer(f'w_{i}').to_dense(),
                    self.get_buffer(f'b_{i}').to_dense(),
                    stride=self.strides,
                    padding=self.padding
                )
            )
        return outputs
