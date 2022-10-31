import numpy as np
import torch


class MultiMaskedDense(torch.nn.Module):
    """
    Multi-Fully-Connected layer which supports masking and pruning
    """

    def __init__(
        self,
        in_features,
        out_features,
        num_tasks,
        device=None,
        dtype=None
    ):
        """
        Parameters
        ----------
        in_features : int
            The number of input features
        out_features : int
            The number of output features.
            Also known as the number of artificial neurons
        num_tasks : int
            The number of tasks to initialize for
        """

        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.num_tasks = num_tasks

        weight = torch.Tensor(
            num_tasks,
            in_features,
            out_features
        ).to(**factory_kwargs)
        weight = torch.nn.init.kaiming_normal_(weight, a=np.sqrt(5))
        self.w = torch.nn.Parameter(weight)
        self.register_buffer(
            'w_mask', torch.ones_like(self.w, **factory_kwargs))

        bias = torch.zeros(num_tasks, out_features, **factory_kwargs)
        self.b = torch.nn.Parameter(bias)
        self.register_buffer('b_mask', torch.ones_like(bias, **factory_kwargs))

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
            out = torch.mm(inputs[i], self.w[i] * self.w_mask[i])
            out = torch.add(out, self.b[i] * self.b_mask[i])
            outputs.append(out)
        return outputs

    def prune(self, percentile):
        """
        Prune the layer by updating the layer's mask

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
            self.w.detach() * self.w_mask
        )
        self.b = torch.nn.Parameter(
            self.b.detach() * self.b_mask
        )
