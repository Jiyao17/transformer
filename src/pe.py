
import torch
import torch.nn as nn

# an implementation of positional encoding

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, device='cpu'):
        super(PositionalEncoding, self).__init__()
        assert d_model % 2 == 0, 'd_model should be an even number'
        
        # shape of pe: (max_len, d_model)
        pe = torch.zeros(max_len, d_model, device=device)
        # shape of position: (max_len, 1)
        # content of position: [[0], [1], [2], ..., [max_len-1]]
        position = torch.arange(0, max_len, device=device).unsqueeze(1)
        # shape of div_term: (d_model/2)
        # shape of i: (d_model/2)
        # content of i: [0, 1, 2, ..., d_model//2]
        i = torch.arange(0, d_model//2, device=device)
        # shape of div_term: (d_model//2)
        # e^(2* i * log(10000) / d_model) = (e^log(10000))^(2 * i / d_model) 
        # = 10000^(2 * i / d_model)
        div_term = torch.exp(torch.log(torch.tensor(10000.0)) * (2 * i / d_model))
        # shape of pe: (max_len, d_model)
        # pe[:, 0::2]: all even columns of pe
        pe[:, 0::2] = torch.sin(position / div_term)
        # pe[:, 1::2]: all odd columns of pe
        pe[:, 1::2] = torch.cos(position / div_term)
        
        # shape of pe: (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # shape of x: (batch_size, seq_len, d_model)
        # shape of self.pe: (1, max_len, d_model)
        # shape of x + self.pe[:, :seq_len]: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]

    def __call__(self, *args, **kwds) -> torch.Tensor:
        return self.forward(*args, **kwds)

# test
if __name__ == '__main__':
    d_model = 512
    max_len = 1024
    batch_size = 32
    pe = PositionalEncoding(d_model, device='cuda')
    x = torch.zeros(batch_size, max_len, d_model, device='cuda')
    y = pe(x)
    print(y.shape)

    print(pe.pe[:, :10, :32])