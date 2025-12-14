import torch 
import torch.nn as nn
from .trf_block.trf import TransformerBlock
from .trf_block.layerNorm.layer_norm import LayerNorm

class GPTModel(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg['vocab_size'],cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg['context_length'],cfg['emb_dim'])
        self.drop_emb = nn.Dropout(cfg['drop_rate'])
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg['n_layers'])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg['emb_dim'],cfg['vocab_size'],bias = False)

    def forward(self,x):
        #shape (b,seq_len)
        batch_size , seq_len = x.shape
        tok_embs = self.tok_emb(x) # (b,seq_len,emb_dim)
        pos_embs = self.pos_emb(torch.arange(seq_len,device=x.device)) #By default, torch.arange creates a tensor on the CPU, even if your model and input are on the GPU.
        x = tok_embs + pos_embs # (b,seq_len,emb_dim)
        x = self.drop_emb(x) # (b,seq_len,emb_dim)
        x = self.trf_blocks(x) # (b,seq_len,emb_dim)
        x = self.final_norm(x) # (b,seq_len,emb_dim)
        logits = self.out_head(x) # (b,seq_len,vocab_size)
        return logits