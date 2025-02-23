
import torch
import torch.nn as nn
import torch.nn.functional as F

class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float=0.1) -> None:
        super(FFN, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, d_model)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    

if __name__ == '__main__':
    batch_size = 64
    seq_len = 128
    d_model = 512
    d_ff = 2048
    x = torch.randn(batch_size, seq_len, d_model)
    ffn = FFN(d_model, d_ff)
    y = ffn(x)
    print(y.shape)