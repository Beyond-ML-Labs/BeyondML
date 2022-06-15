import torch

class MultiDense(torch.nn.Module):

    def __init__(
            self,
            weight,
            bias
    ):
        super().__init__()
        self.w = torch.Tensor(weight).to_sparse()
        self.b = torch.Tensor(bias).to_sparse()

    def forward(self, inputs):
        outputs = []
        for i in range(len(inputs)):
            out = torch.sparse.mm(self.w[i].t(), inputs[i].t()).t()
            out = torch.add(out, self.b[i])
            outputs.append(out)
        return outputs
