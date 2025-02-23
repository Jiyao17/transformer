
import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        
        out = self.gamma * (x - mean) / (std + self.eps) + self.beta
        return out
    

if __name__ == '__main__':

    batch_size = 64
    seq_len = 128
    d_model = 512
    x = torch.randn(batch_size, seq_len, d_model)
    ln = LayerNorm(d_model)
    y = ln(x)
    print(y.shape)
    print(y[0, 0, :10])