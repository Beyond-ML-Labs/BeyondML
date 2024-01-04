import torch
import math


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float = 0.1):
        """
        Parameters
        ----------
        d_model : int
            The dimensionality of the model
        seq_len : int
            The maximum allowed sequence length
        dropout : float (default 0.1)
            The dropout rate for the encoding layer
        """

        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = dropout

        self.dropout_layer = torch.nn.Dropout(dropout)

        pe = torch.zeros(self.seq_len, self.d_model)
        position = torch.arange(
            0, self.seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / self.d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x : torch.tensor):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout_layer(x)
