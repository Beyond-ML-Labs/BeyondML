import torch


class SparseMultiDense(torch.nn.Module):
    """
    Sparse implementation of the Multi-Fully-Connected layer
    """

    def __init__(
            self,
            weight,
            bias
    ):
        """
        Parameters
        ----------
        weight : torch.Tensor or Tensor-like
            The weight to use
        bias : torch.Tensor or Tensor-like
            The bias to use
        """

        super().__init__()
        self.w = torch.Tensor(weight).to_sparse()
        self.b = torch.Tensor(bias).to_sparse()

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
            out = torch.sparse.mm(self.w[i].t(), inputs[i].t()).t()
            out = torch.add(out, self.b[i].to_dense())
            outputs.append(out)
        return outputs
