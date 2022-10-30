import torch
from typing import Optional, Any, Union, Callable
from torch.nn import functional as F
from torch import Tensor
from torch.nn import Dropout, LayerNorm
from beyondml.pt.layers import MaskedDense
from .MaskedMultiHeadAttention import MaskedMultiHeadAttention


class MaskedTransformerDecoderLayer(torch.nn.Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to self attention, multihead
            attention and feedforward operations, respectively. Otherwise it's done after.
            Default: ``False`` (after).
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False,
                 norm_first: bool = False,
                 device=None,
                 dtype=None
                 ) -> None:

        factory_kwargs = {'device': device, 'dtype': dtype}

        super(MaskedTransformerDecoderLayer, self).__init__()

        self.self_attn = MaskedMultiHeadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            **factory_kwargs
        )
        self.multihead_attn = MaskedMultiHeadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            **factory_kwargs
        )

        # Implementation of Feedforward model
        self.linear1 = MaskedDense(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = MaskedDense(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(MaskedTransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor):
        r"""Pass the inputs (and mask) through the decoder layer.
        Args:
            tgt: the sequence to the decoder layer.
            memory: the sequence from the last layer of the encoder.
        Shape:
            see the docs in Pytorch Transformer class.
        """

        x = tgt
        x = self._sa_block(x, memory)
        x = self._mha_block(x, memory)
        x = self._ff_block(x, memory)

    # self-attention block

    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

    def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu

        raise RuntimeError(
            "activation should be relu/gelu, not {}".format(activation))

    def prune(self, percentile):
        self.self_attn.prune(percentile)
        self.multihead_attn.prune(percentile)
        self.linear1.prune(percentile)
        self.linear2.prune(percentile)
