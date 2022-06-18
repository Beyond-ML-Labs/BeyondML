import torch

class Dense(torch.nn.Module):

    def __init__(
        self,
        weight,
        bias
    ):
        super().__init__()
        self.w = torch.nn.Parameter(torch.Tensor(weight))
        self.b = torch.nn.Parameter(torch.Tensor(bias))

    def forward(self, inputs):
        out = torch.mm(inputs, self.w)
        out = torch.add(out, self.b)
        return out
