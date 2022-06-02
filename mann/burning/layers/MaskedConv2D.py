import numpy as np
import torch

class MaskedConv2D(torch.nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size = 3,
        padding = 'same',
        strides = 1
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = strides

        filters = torch.Tensor(
            self.out_channels,
            self.in_channels,
            self.kernel_size[0],
            self.kernel_size[1]
        )
        filters = torch.nn.init.normal_(filters)
        self.w = torch.nn.Parameter(filters)
        self.w_mask = torch.ones_like(self.w)

        bias = torch.zeros(out_channels)
        self.b = torch.nn.Parameter(bias)
        self.b_mask = torch.ones_like(self.b)
        

    @property
    def in_channels(self):
        return self._in_channels
    @in_channels.setter
    def in_channels(self, value):
        if not isinstance(value, int):
            raise TypeError('in_channels must be int')
        self._in_channels = value

    @property
    def out_channels(self):
        return self._out_channels
    @out_channels.setter
    def out_channels(self, value):
        if not isinstance(value, int):
            raise TypeError('out_channels must be int')
        self._out_channels = value
    
    @property
    def kernel_size(self):
        return self._kernel_size
    @kernel_size.setter
    def kernel_size(self, value):
        if isinstance(value, int):
            value = (value, value)
        elif isinstance(value, tuple):
            if not all([isinstance(val, int) for val in value]) and len(value) == 2:
                raise ValueError('If tuple, kernel_size must be two integers')
        else:
            raise TypeError('kernel_size must be int or tuple')
        self._kernel_size = value
    
    def forward(self, inputs):
        return torch.nn.functional.conv2d(
                inputs,
                self.w * self.w_mask,
                self.b * self.b_mask,
                stride = self.strides,
                padding = self.padding
            )

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
