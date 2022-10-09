import torch

class MultiConv3D(torch.nn.Module):

    def __init__(
        self,
        kernel,
        bias,
        padding = 'same',
        strides = 1
    ):

        self.w = torch.nn.Parameter(
            torch.Tensor(kernel)
        )
        self.b = torch.nn.Parameter(
            torch.Tensor(bias)
        )

        self.padding = padding
        self.strides = strides

    def forward(
        self,
        inputs
    ):

        outputs = []
        for i in range(len(inputs)):
            outputs.append(
                torch.nn.functional.conv3d(
                    inputs[i],
                    self.w[i],
                    self.b[i],
                    stride = self.strides,
                    padding = self.padding
                )
            )
        return outputs
