import torch
import torch.nn as nn
class LayerNorm(nn.Module):
    def __init__(self,emb_dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
        self.eps = 1e-5

    def forward(self,x):
        #inp shape => (b,seq_len,emb_out)
        mean = x.mean(dim = -1,keepdim = True) #(b,1)
        std = x.std(dim=-1,keepdim = True ,unbiased = False) #(b,1)
        norm_x  = (x - mean)/(std + self.eps) #(b,seq_len,emb_out)
        return norm_x * self.scale + self.shift #(b,seq_len,emb_out)