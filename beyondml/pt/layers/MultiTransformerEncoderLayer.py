import torch 
from typing import Optional, Any, Union, Callable
import torch
from torch import Tensor
from torch.nn import Dropout, LayerNorm
from beyondml.pt.layers import  MultiMaskedDense

class MultiTransformerEncoderLayer(torch.nn.Module):
    """MultiTransformerEncoderLayer is made up of self-attn and feedforward network.
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
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).
        num_tasks: number of tasks to initialize for. Default: 1.
        percentile: the int between 0 and 99 representing the proportion of weights in the
            feedforward block to be inactive
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = torch.nn.functional.relu,
                 layer_norm_eps: float = 1e-5,
                 batch_first: bool = False,
                 norm_first: bool = False,
                 num_tasks: int = 1,
                 percentile: int = 0,
                 device=None, 
                 dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MultiTransformerEncoderLayer, self).__init__()
        self.self_attn = torch.nn.MultiheadAttention(d_model, 
                                                    nhead,
                                                    dropout=dropout,
                                                    batch_first=batch_first,
                                                    **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = MultiMaskedDense(d_model, dim_feedforward, num_tasks)
        self.dropout = Dropout(dropout)
        self.linear2 = MultiMaskedDense(d_model, dim_feedforward, num_tasks)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)
        if activation is torch.nn.functional.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is torch.nn.functional.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        super(MultiTransformerEncoderLayer, self).__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu

    def forward(self, src: Tensor, 
                percentile: int = 0, 
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
       
        x = src
        x = self._sa_block(x)
        x = self._ff_block(x)
        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
    
    def prune(self, percentile):
        self.linear1(percentile)
        self.linear2(percentile)