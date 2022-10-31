from .MaskedDense import MaskedDense
import numpy as np
import torch


class MaskedMultiHeadAttention(torch.nn.Module):
    """
    Masked Multi-Headed Attention Layer
    """

    def __init__(
            self,
            embed_dim,
            num_heads,
            dropout=0,
            batch_first=False,
            device=None,
            dtype=None
    ):
        """
        Parameters
        ----------
        embed_dim : int
            The embedding dimension
        num_heads : int
            The number of attention heads
        dropout : float (default 0)
            The dropout rate to apply
        batch_first : bool (default False)
            Whether the batch dimension is first
        """

        super().__init__()

        factory_kwargs = {'device': device, 'dtype': dtype}
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first

        self.head_dim = embed_dim // num_heads

        if self.head_dim * self.num_heads != embed_dim:
            raise ValueError('num_heads must evenly divide embed_dim')

        in_proj_weight = torch.Tensor(
            3 * embed_dim, embed_dim).to(**factory_kwargs)
        in_proj_weight = torch.nn.init.xavier_uniform_(in_proj_weight)
        self.in_proj_weight = torch.nn.Parameter(in_proj_weight)
        self.register_buffer('in_proj_weight_mask', torch.ones_like(
            self.in_proj_weight, **factory_kwargs))

        self.in_proj_bias = torch.nn.Parameter(
            torch.zeros((3 * embed_dim), **factory_kwargs))
        self.register_buffer('in_proj_bias_mask', torch.ones_like(
            self.in_proj_bias, **factory_kwargs))

        self.out_proj = MaskedDense(
            embed_dim, embed_dim, **factory_kwargs)
        self.out_proj_weight = self.out_proj.w
        self.out_proj_weight_mask = self.out_proj.w_mask
        self.out_proj_bias = self.out_proj.b
        self.out_proj_bias_mask = self.out_proj.b_mask

    def forward(
            self,
            query,
            key,
            value,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True
    ):
        """
        Call the layer on input data

        Parameters
        ----------
        query : torch Tensor
            Query tensor
        key : torch Tensor
            Key tensor
        value : torch Tensor
            Value tensor
        key_padding_mask : None or torch Tensor (default None)
            If specified, a mask indicating which elements in ``key`` to ignore
        need_weights : Bool (default True)
            If specified, returns ``attn_output_weights`` as well as ``attn_outputs``
        attn_mask : None or torch Tensor (default None)
            If specified, a 2D or 3D mask preventing attention
        average_attn_weights : Bool (default True)
            If True, indicates that returned ``attn_weights`` should be averaged across heads
        """

        is_batched = query.dim() == 3

        if self.batch_first and is_batched:
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(
                    1, 0) for x in (query, key, value)]

        attn_output, attn_output_weights = torch.nn.functional.multi_head_attention_forward(
            query,
            key,
            value,
            self.embed_dim,
            self.num_heads,
            self.in_proj_weight * self.in_proj_weight_mask,
            self.in_proj_bias * self.in_proj_bias_mask,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights
        )

        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights

    def prune(self, percentile):
        """
        Prune the layer by updating the layer's mask

        Parameters
        ----------
        percentile : int
            Integer between 0 and 99 which represents the proportion of weights to be made inactive

        Notes
        -----
        Acts on the layer in place
        """
        w_copy = np.abs(self.in_proj_weight.detach().cpu().numpy())
        b_copy = np.abs(self.in_proj_bias.detach().cpu().numpy())
        w_percentile = np.percentile(w_copy, percentile)
        b_percentile = np.percentile(b_copy, percentile)

        new_w_mask = torch.Tensor(
            (w_copy >= w_percentile).astype(int))
        new_b_mask = torch.Tensor(
            (b_copy >= b_percentile).astype(int))
        self.in_proj_weight_mask[:] = new_w_mask
        self.in_proj_bias_mask[:] = new_b_mask

        self.in_proj_weight = torch.nn.Parameter(
            self.in_proj_weight * self.in_proj_weight_mask
        )
        self.in_proj_bias = torch.nn.Parameter(
            self.in_proj_bias * self.in_proj_bias_mask
        )
        self.out_proj.prune(percentile)
