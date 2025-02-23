
import torch
from torch import nn
import torch.nn.functional as F


class ScaleDotProdAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProdAttention, self).__init__()

    def forward(self, 
            q: torch.Tensor, 
            k: torch.Tensor, 
            v: torch.Tensor, 
            mask: torch.Tensor = None
            ) -> torch.Tensor:
        # q: (batch_size, n_heads, seq_len, d_k)
        # k: (batch_size, n_heads, seq_len, d_k)
        # v: (batch_size, n_heads, seq_len, d_v)
        # mask: (batch_size, n_heads, seq_len, seq_len)
        # attn: (batch_size, n_heads, seq_len, seq_len)
        d_k = q.size(-1)

        # shape of attn: (batch_size, n_heads, seq_len, seq_len)
        attn = torch.matmul(q, k.transpose(-1, -2)) / torch.sqrt(torch.tensor(d_k).float())

        # mask is used to prevent attention to padding tokens
        if mask is not None:
            mask = mask.unsqueeze(1)
            attn = attn.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(attn, dim=-1)
        # shape of output: (batch_size, n_heads, seq_len, d_v)
        output = torch.matmul(attn, v)
        
        return output
    

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads: int, d_model: int, d_k: int=None, d_v: int=None) -> None:
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_k if d_k is not None else d_model
        self.d_v = d_v if d_v is not None else d_model

        assert d_model % n_heads == 0, 'd_model should be divisible by n_heads'
        
        self.w_q = nn.Linear(d_model, n_heads * self.d_k)
        self.w_k = nn.Linear(d_model, n_heads * self.d_k)
        self.w_v = nn.Linear(d_model, n_heads * self.d_v)
        self.attention = ScaleDotProdAttention()
        self.fc = nn.Linear(n_heads * self.d_v, d_model)

    def forward(self, 
            q: torch.Tensor, 
            k: torch.Tensor, 
            v: torch.Tensor, 
            mask: torch.Tensor=None
            ) -> torch.Tensor:
        # q: (batch_size, seq_len, d_model)
        # k: (batch_size, seq_len, d_model)
        # v: (batch_size, seq_len, d_model)
        # mask: (batch_size, seq_len, seq_len)
        batch_size, seq_len, _ = q.size()
        
        # shape of q: (batch_size, seq_len, n_heads * d_k)
        q = self.w_q(q).view(batch_size, seq_len, self.n_heads, self.d_k)
        # shape of k: (batch_size, seq_len, n_heads * d_k)
        k = self.w_k(k).view(batch_size, seq_len, self.n_heads, self.d_k)
        # shape of v: (batch_size, seq_len, n_heads * d_v)
        v = self.w_v(v).view(batch_size, seq_len, self.n_heads, self.d_v)
        
        # shape of q: (batch_size, n_heads, seq_len, d_k)
        q = q.permute(0, 2, 1, 3)
        # shape of k: (batch_size, n_heads, seq_len, d_k)
        k = k.permute(0, 2, 1, 3)
        # shape of v: (batch_size, n_heads, seq_len, d_v)
        v = v.permute(0, 2, 1, 3)
        
        # shape of output: (batch_size, n_heads, seq_len, d_v)
        output = self.attention(q, k, v, mask)
        # shape of output: (batch_size, seq_len, n_heads * d_v)
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)
        # shape of output: (batch_size, seq_len, d_model)
        output = self.fc(output)
        
        return output
    
    def __call__(self, *args, **kwds) -> torch.Tensor:
        return self.forward(*args, **kwds)


# test
if __name__ == '__main__':
    batch_size = 32
    seq_len = 1024
    d_model = 512
    n_heads = 8
    d_k = d_v = d_model // n_heads
    x = torch.randn(batch_size, seq_len, d_model)
    mask = torch.ones(batch_size, seq_len, seq_len)
    mask[:, -seq_len//2:] = 0
    
    mha = MultiHeadAttention(n_heads, d_model, d_k, d_v)
    y = mha(x, x, x, mask)
    print(y.shape)
    print(y[0, 0, :10])