import torch


class SparseMultiDense(torch.nn.Module):
    """
    Sparse implementation of the Multi-Fully-Connected layer
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
            The weight to use
        bias : torch.Tensor or Tensor-like
            The bias to use
        """

        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        for i in range(weight.shape[0]):
            self.register_buffer(
                f'w_{i}',
                torch.Tensor(weight[i]).to(**factory_kwargs).to_sparse()
            )
            self.register_buffer(
                f'b_{i}',
                torch.Tensor(bias[i]).to(**factory_kwargs).to_sparse()
            )

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
            out = torch.sparse.mm(
                self.get_buffer(f'w_{i}').t(),
                inputs[i].t()
            ).t()
            out = torch.add(
                out,
                self.get_buffer(f'b_{i}').to_dense()
            )
            outputs.append(out)
        return outputs
