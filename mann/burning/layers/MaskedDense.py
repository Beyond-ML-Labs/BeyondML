import torch

class MaskedDense(torch.nn.Module):

    def __init__(
        self,
        in_features,
        out_features,
        use_bias = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features,
        self.use_bias = use_bias

        weight = torch.Tensor(
            in_features,
            out_features
        )
        weight = torch.nn.init.normal_(weight)
        self.w = torch.nn.Parameter(weight)
        self.w_mask = torch.ones_like(self.w)

        if self.use_bias:
            bias = torch.zeros(out_features)
            self.b = torch.nn.Parameter(bias)
            self.b_mask = torch.ones_like(bias)

    def forward(self, inputs):
        out = torch.mm(inputs, self.w * self.w_mask)
        if self.use_bias:
            out = torch.add(out, self.b * self.b_mask)
        return out
