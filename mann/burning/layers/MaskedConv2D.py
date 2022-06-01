import torch

class MaskedConv2D(torch.nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size = 3,
        padding = 'same',
        strides = 1,
        use_bias = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = strides
        self.use_bias = use_bias

        filters = torch.Tensor(
            self.out_channels,
            self.in_channels,
            self.kernel_size[0],
            self.kernel_size[1]
        )
        filters = torch.nn.init.normal_(filters)
        self.w = torch.nn.Parameter(filters)
        self.w_mask = torch.ones_like(self.weight)

        if self.use_bias:
            bias = torch.zeros(filters)
            bias = torch.nn.Parameter(bias)
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
        conv = torch.nn.functional.conv2d(
            inputs,
            self.w * self.w_mask,
            stride = self.strides,
            padding = self.padding
        )
        if self.use_bias:
            conv = torch.add(conv, self.b * self.b_mask)
        
        return conv
