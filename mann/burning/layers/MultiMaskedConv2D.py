import numpy as np
import torch

class MultiMaskedConv2D(torch.nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        num_tasks,
        kernel_size = 3,
        padding = 'same',
        strides = 1
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_tasks = num_tasks
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = strides

        filters = torch.Tensor(
            self.num_tasks,
            self.out_channels,
            self.in_channels,
            self.kernel_size[0],
            self.kernel_size[1]
        )
        filters = torch.nn.init.normal_(filters)
        self.w = torch.nn.Parameter(filters)
        self.w_mask = torch.ones_like(self.w)

        bias = torch.zeros(self.num_tasks, out_channels)
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
        outputs = []
        for i in range(len(inputs)):
            outputs.append(
                torch.nn.functional.conv2d(
                    inputs[i],
                    self.w[i] * self.w_mask[i],
                    self.b[i] * self.b_mask[i],
                    stride = self.strides,
                    padding = self.padding
                )
            )

    def prune(self, percentile):
        
        w_copy = np.abs(self.w.detach().numpy())
        b_copy = np.abs(self.b.detach().numpy())
        new_w_mask = np.zeros_like(w_copy)
        new_b_mask = np.zeros_like(b_copy)

        for task_num in range(self.num_tasks):
            if task_num != 0:
                for prev_idx in range(task_num - 1):
                    w_copy[task_num][new_w_mask[prev_idx] == 1] = 0
                    b_copy[task_num][new_b_mask[prev_idx] == 1] = 0
            
            w_percentile = np.percentile(w_copy[task_num], percentile)
            b_percentile = np.percentile(b_copy[task_num], percentile)

            new_w_mask[task_num] = (w_copy[task_num] >= w_percentile).astype(int)
            new_b_mask[task_num] = (b_copy[task_num] >= b_percentile).astype(int)

        self.w_mask = torch.Tensor(new_w_mask)
        self.b_mask = torch.Tensor(new_b_mask)

        self.w = torch.nn.Parameter(
            self.w * self.w_mask
        )
        self.b = torch.nn.Parameter(
            self.b * self.b_mask
        )
