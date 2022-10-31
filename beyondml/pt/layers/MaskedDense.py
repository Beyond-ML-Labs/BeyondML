import numpy as np
import torch


class MaskedDense(torch.nn.Module):
    """
    Masked fully-connected layer
    """

    def __init__(
        self,
        in_features,
        out_features,
        device=None,
        dtype=None
    ):
        """
        Parameters
        ----------
        in_features : int
            The number of features input to the layer
        out_features : int
            The number of features to be output by the layer.
            Also considered the number of artificial neurons
        """

        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_features = in_features
        self.out_features = out_features

        weight = torch.Tensor(
            in_features,
            out_features,
        ).to(**factory_kwargs)
        weight = torch.nn.init.kaiming_normal_(weight, a=np.sqrt(5))
        self.w = torch.nn.Parameter(weight)
        self.register_buffer(
            'w_mask', torch.ones_like(self.w, **factory_kwargs))

        bias = torch.zeros(out_features, **factory_kwargs)
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
        weight = self.w * self.w_mask
        bias = self.b * self.b_mask
        out = torch.mm(inputs, weight)
        out = torch.add(out, bias)
        return out

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
        w_percentile = np.percentile(w_copy, percentile)
        b_percentile = np.percentile(b_copy, percentile)

        new_w_mask = torch.Tensor(
            (w_copy >= w_percentile).astype(int))
        new_b_mask = torch.Tensor(
            (b_copy >= b_percentile).astype(int))
        self.w_mask[:] = new_w_mask
        self.b_mask[:] = new_b_mask

        self.w = torch.nn.Parameter(
            self.w.detach() * self.w_mask
        )
        self.b = torch.nn.Parameter(
            self.b.detach() * self.b_mask
        )
