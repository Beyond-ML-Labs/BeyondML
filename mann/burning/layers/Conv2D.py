import torch


class Conv2D(torch.nn.Module):

    def __init__(
        self,
        kernel,
        bias,
        padding='same',
        strides=1
    ):
        super().__init__()
        self.w = torch.nn.Parameter(torch.Tensor(kernel))
        self.b = torch.nn.Parameter(torch.Tensor(bias))

        self.padding = padding
        self.strides = strides

    def forward(
        self,
        inputs
    ):

        return torch.nn.functional.conv2d(
            inputs,
            self.w,
            self.b,
            stride=self.strides,
            padding=self.padding
        )
