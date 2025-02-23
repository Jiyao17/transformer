
import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import MultiHeadAttention
from layernorm import LayerNorm
from ffn import FFN
from pe import PositionalEncoding


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ffn, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.attn = MultiHeadAttention(n_heads, d_model)
        self.ln1 = LayerNorm(d_model)
        self.ffn = FFN(d_model, d_ffn, dropout)
        self.ln2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn = self.attn(x, x, x, mask)
        attn = self.ln1(x + attn)

        ffn = self.ffn(attn)
        ffn = self.ln2(attn + ffn)

        out = self.dropout(ffn)

        return out
    

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ffn, n_layers, dropout):
        super(TransformerEncoder, self).__init__()

        self.pe = PositionalEncoding(d_model)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ffn, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x, mask):
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x
    

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    seq_len = 512
    d_model = 64
    n_heads = 8
    d_ffn = 256
    n_layers = 8
    dropout = 0.1

    x = torch.randn(batch_size, seq_len, d_model, device=device)
    mask = torch.ones(batch_size, seq_len, seq_len, device=device)
    mask[:, -seq_len//2:] = 0

    encoder = TransformerEncoder(d_model, n_heads, d_ffn, n_layers, dropout)
    encoder = encoder.to(device)
    y = encoder(x, mask)
    print(y.shape)
    print(y[0, 0, :10])