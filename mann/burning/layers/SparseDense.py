import torch


class SparseDense(torch.nn.Module):
    """
    Sparse implementation of a fully-connected layer
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
        out = torch.sparse.mm(self.w.t(), inputs.t()).t()
        out = torch.add(out, self.b.to_dense())
        return out
