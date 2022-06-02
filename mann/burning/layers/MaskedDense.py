import numpy as np
import torch

class MaskedDense(torch.nn.Module):

    def __init__(
        self,
        in_features,
        out_features
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        weight = torch.Tensor(
            in_features,
            out_features
        )
        weight = torch.nn.init.normal_(weight)
        self.w = torch.nn.Parameter(weight)
        self.w_mask = torch.ones_like(self.w)

        bias = torch.zeros(out_features)
        self.b = torch.nn.Parameter(bias)
        self.b_mask = torch.ones_like(bias)

    def forward(self, inputs):
        out = torch.mm(inputs, self.w * self.w_mask)
        out = torch.add(out, self.b * self.b_mask)
        return out

    def prune(self, percentile):
        w_copy = np.abs(self.w.detach().numpy())
        b_copy = np.abs(self.b.detach().numpy())
        w_percentile = np.percentile(w_copy, percentile)
        b_percentile = np.percentile(b_copy, percentile)
        
        new_w_mask = torch.Tensor((w_copy >= w_percentile).astype(int))
        new_b_mask = torch.Tensor((b_copy >= b_percentile).astype(int))
        self.w_mask = new_w_mask
        self.b_mask = new_b_mask

        self.w = torch.nn.Parameter(
            self.w * self.w_mask
        )
        self.b = torch.nn.Parameter(
            self.b * self.b_mask
        )
