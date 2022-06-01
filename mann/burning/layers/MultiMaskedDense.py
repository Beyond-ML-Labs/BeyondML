import numpy as np
import torch

class MultiMaskedDense(torch.nn.Module):

    def __init__(
        self,
        in_features,
        out_features,
        num_tasks,
        use_bias = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_tasks = num_tasks
        self.use_bias = use_bias

        weight = torch.Tensor(
            num_tasks,
            in_features,
            out_features
        )
        weight = torch.nn.init.normal_(weight)
        self.w = torch.nn.Parameter(weight)
        self.w_mask = torch.ones_like(self.w)

        if self.use_bias:
            bias = torch.zeros(num_tasks, out_features)
            self.b = torch.nn.Parameter(bias)
            self.b_mask = torch.ones_like(bias)

    def forward(self, inputs):
        outputs = []
        for i in range(len(inputs)):
            out = torch.mm(inputs[i], self.w[i] * self.w_mask[i])
            if self.use_bias:
                out = torch.add(out, self.b[i] * self.b_mask[i])
            outputs.append(out)
        return outputs

    def prune(self, percentile):
        raise NotImplementedError
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
