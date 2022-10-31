import numpy as np
import torch


class MultiMaskedConv3D(torch.nn.Module):
    """
    Masked Multitask 3D Convolutional layer
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_tasks,
        kernel_size=3,
        padding='same',
        strides=1,
        device=None,
        dtype=None
    ):
        """
        Parameters
        ----------
        in_channels : int
            The number of channels for input data
        out_channels : int
            The number of filters to use
        kernel_size : int or tuple (default 3)
            The kernel size to use
        padding : int or str (default 'same')
            Padding to use
        strides : int or tuple (default 1)
            The number of strides to use
        """

        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
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
            self.kernel_size[1],
            self.kernel_size[2]
        ).to(**factory_kwargs)
        filters = torch.nn.init.kaiming_normal_(filters, a=np.sqrt(5))
        self.w = torch.nn.Parameter(filters)
        self.register_buffer(
            'w_mask', torch.ones_like(self.w, **factory_kwargs))

        bias = torch.zeros(self.num_tasks, out_channels, **factory_kwargs)
        self.b = torch.nn.Parameter(bias)
        self.register_buffer(
            'b_mask', torch.ones_like(self.b, **factory_kwargs))

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
            value = (value, value, value)
        elif isinstance(value, tuple):
            if not all([isinstance(val, int) for val in value]) and len(value) == 3:
                raise ValueError(
                    'If tuple, kernel_size must be three integers')
        else:
            raise TypeError('kernel_size must be int or tuple')
        self._kernel_size = value

    def forward(self, inputs):
        """
        Call the layer on input data

        Parameters
        ----------
        inputs : torch.Tensor
            Inputs to call the layer's logic on

        Returns
        -------
        results : torch.Tensor
            The results of the layer's logic
        """
        outputs = []
        for i in range(len(inputs)):
            outputs.append(
                torch.nn.functional.conv3d(
                    inputs[i],
                    self.w[i] * self.w_mask[i],
                    self.b[i] * self.b_mask[i],
                    stride=self.strides,
                    padding=self.padding
                )
            )
        return outputs

    def prune(self, percentile):
        """
        Prune the layer by updating the layer's masks

        Parameters
        ----------
        percentile : int
            Integer between 0 and 99 which represents the proportion of weights to be inactive

        Notes
        -----
        Acts on the layer in place
        """

        w_copy = np.abs(self.w.detach().cpu().numpy())
        b_copy = np.abs(self.b.detach().cpu().numpy())
        new_w_mask = np.zeros_like(w_copy)
        new_b_mask = np.zeros_like(b_copy)

        for task_num in range(self.num_tasks):
            if task_num != 0:
                for prev_idx in range(task_num):
                    w_copy[task_num][new_w_mask[prev_idx] == 1] = 0
                    b_copy[task_num][new_b_mask[prev_idx] == 1] = 0

            w_percentile = np.percentile(w_copy[task_num], percentile)
            b_percentile = np.percentile(b_copy[task_num], percentile)

            new_w_mask[task_num] = (
                w_copy[task_num] >= w_percentile).astype(int)
            new_b_mask[task_num] = (
                b_copy[task_num] >= b_percentile).astype(int)

            self.w_mask[:] = torch.Tensor(new_w_mask)
            self.b_mask[:] = torch.Tensor(new_b_mask)

            self.w = torch.nn.Parameter(
                self.w * self.w_mask
            )
            self.b = torch.nn.Parameter(
                self.b * self.b_mask
            )
