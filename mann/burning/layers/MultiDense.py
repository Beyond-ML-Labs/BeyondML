import torch

class MultiDense(torch.nn.Module):

    def __init__(
            self,
            weight,
            bias
    ):
        super().__init__()
        self.w = torch.nn.Parameter(torch.Tensor(weight))
        self.b = torch.nn.Parameter(torch.Tensor(bias))

    def forward(self, inputs):
        outputs = []
        for i in range(len(inputs)):
            out = torch.mm(inputs[i], self.w[i])
            out = torch.add(out, self.b[i])
            outputs.append(out)
        return outputs
