import torch


class SparseMultiConv2D(torch.nn.Module):

    def __init__(
            self,
            kernel,
            bias,
            padding='same',
            strides=1
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
        outputs = []
        kernel = self.w.to_dense()
        bias = self.b.to_dense()

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
