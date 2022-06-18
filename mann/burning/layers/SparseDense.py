import torch

class SparseDense(torch.nn.Module):

    def __init__(
        self,
        weight,
        bias
    ):
        super().__init__()
        self.w = torch.Tensor(weight).to_sparse()
        self.b = torch.Tensor(bias).to_sparse()

    def forward(self, inputs):
        out = torch.sparse.mm(self.w.t(), inputs.t()).t()
        out = torch.add(out, self.b)
        return out
