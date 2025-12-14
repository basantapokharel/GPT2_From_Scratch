import torch 
import torch.nn as nn

class FeedForwardNetwork(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg['emb_dim'],4 * cfg['emb_dim']),
            nn.GELU(),
            nn.Linear(4 * cfg['emb_dim'],cfg['emb_dim'])
        )
        

    def forward(self,x):
        #shape(b,seq_len,emb_dim)
        return self.layers(x)#output shape also same:(batch_size,seq_len,emb_dim)
        