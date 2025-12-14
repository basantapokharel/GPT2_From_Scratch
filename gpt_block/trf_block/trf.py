import torch 
import torch.nn as nn
from .attention.masked_mha import MultiHeadAttention
from .feed_forward_net.ffn import FeedForwardNetwork
from .layerNorm.layer_norm import LayerNorm



class TransformerBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.norm1 = LayerNorm(cfg['emb_dim'])
        self.att = MultiHeadAttention(cfg['emb_dim'],cfg['emb_dim'],cfg['context_length'],cfg['drop_rate'],cfg['n_heads'],cfg['qkv_bias'])
        self.drop_shortcut = nn.Dropout(cfg['drop_rate'])
        self.norm2 = LayerNorm(cfg['emb_dim'])
        self.ff = FeedForwardNetwork(cfg)

    def forward(self,x):
        #input shape(b,seq_len,emb_dim)
        shortcut = x
        x = self.norm1(x)
        x = self.att(x) #(b,seq_len,emb_dim)
        x = self.drop_shortcut(x)
        x += shortcut

        shortcut = x
        x = self.norm2(x) 
        x = self.ff(x) #(b,seq_len,emb_dim)
        x = self.drop_shortcut(x)
        x += shortcut

        return x  #(b,seq_len,emb_dim)

