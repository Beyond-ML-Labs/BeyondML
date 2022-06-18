import torch

class SparseConv2D(torch.nn.Module):

    def __init__(
        self,
        kernel,
        bias,
        padding = 'same',
        strides = 1
    ):
        super().__init__()
        self.w = torch.Tensor(kernel).to_sparse()
        self.b = torch.Tensor(bias).to_sparse()

        self.padding = padding
        self.strides = strides

    def forward(
        self,
        inputs
    ):
        kernel = self.w.to_dense()
        bias = self.b.to_dense()

        return torch.nn.functional.conv2d(
            inputs,
            kernel,
            bias,
            stride = self.strides,
            padding = self.padding
        )
