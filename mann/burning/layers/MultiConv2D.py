import torch


class MultiConv2D(torch.nn.Module):

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
        outputs = []
        kernel = self.w
        bias = self.b

        for i in range(len(inputs)):
            outputs.append(
                torch.nn.functional.conv2d(
                    inputs[i],
                    kernel[i],
                    bias[i],
                    stride=self.strides,
                    padding=self.padding
                )
            )
        return outputs
