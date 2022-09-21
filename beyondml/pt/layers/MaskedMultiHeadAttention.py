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
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first

        self.head_dim = embed_dim // num_heads

        if self.head_dim * self.num_heads != embed_dim:
            raise ValueError('num_heads must evenly divide embed_dim')

        in_proj_weight = torch.Tensor((3 * embed_dim, embed_dim))
        in_proj_weight = torch.nn.init.xavier_uniform_(in_proj_weight)
        self.in_proj_weight = torch.nn.Parameter(in_proj_weight)
        self.in_proj_mask = torch.ones_like(self.in_proj_weight)

        self.in_proj_bias = torch.nn.Parameter(torch.zeros((3 * embed_dim)))
        self.in_proj_bias_mask = torch.ones_like(self.in_proj_bias)

        self.out_proj = MaskedDense(embed_dim, embed_dim)
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
        w_copy = np.abs(self.in_proj_weight.detach().numpy())
        b_copy = np.abs(self.in_proj_bias.detach().numpy())
        w_percentile = np.percentile(w_copy, percentile)
        b_percentile = np.percentile(b_copy, percentile)

        new_w_mask = torch.Tensor((w_copy >= w_percentile).asypte(int))
        new_b_mask = torch.Tensor((b_copy >= b_percentile).astype(int))
        self.in_proj_weight_mask = new_w_mask
        self.in_proj_bias_mask = new_b_mask

        self.in_proj_weight = torch.nn.Parameter(
            self.in_proj_weight * self.in_proj_mask
        )
        self.in_proj_bias = torch.nn.Parameter(
            self.in_proj_bias * self.in_proj_bias_mask
        )
        self.out_proj.prune(percentile)