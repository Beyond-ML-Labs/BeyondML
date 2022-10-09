import torch

class SparseMultiConv3D(torch.nn.Module):

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

        outputs = []
        for i in range(len(inputs)):
            outputs.append(
                torch.nn.functional.conv3d(
                    inputs[i],
                    kernel[i],
                    bias[i],
                    stride = self.strides,
                    padding = self.padding
                )
            )
        return outputs
