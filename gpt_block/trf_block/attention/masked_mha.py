import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self,d_in,d_out,context_length,dropout,num_heads,qkv_bias=True):
        super().__init__()
        assert (d_out % num_heads == 0),"d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in,d_out,bias = qkv_bias)
        self.W_key = nn.Linear(d_in,d_out,bias = qkv_bias)
        self.W_value = nn.Linear(d_in,d_out,bias = qkv_bias)
        self.register_buffer('mask',torch.triu(torch.ones(context_length,context_length),diagonal=1))
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_out,d_out)
    def forward(self,x):
        # (b,seq_len,d_in)
        b, num_tokens , d_in = x.shape
        keys = self.W_key(x) #(b,seq_len,d_out)
        queries = self.W_query(x) #(batch_size,seq_len,d_out)
        values = self.W_value(x) #(batch_size,seq_len,d_out)

        #unrolling keys , queries and values
        keys = keys.view(b,num_tokens,self.num_heads,self.head_dim)#(batch_size,num_tokens,num_heads,head_dim)
        queries = queries.view(b,num_tokens,self.num_heads,self.head_dim)#(batch_size,num_tokens,num_heads,head_dim)
        values = values.view(b,num_tokens,self.num_heads,self.head_dim)#(batch_size,num_tokens,num_heads,head_dim)

        #every operation is performed on (num_tokens,head_dim) for each batch and num_heads
        keys = keys.transpose(1,2) #(batch_size,num_heads,num_tokens,head_dim)
        queries =queries.transpose(1,2) #(batch_size,num_heads,num_tokens,head_dim)
        values = values.transpose(1,2) #(batch_size,num_heads,num_tokens,head_dim)

        #compute attention scores 
        attn_scores = queries @ keys.transpose(2,3)
        #(batch_size,num_heads,num_tokens,num_tokens)

        #create mask from mask
        mask_bool = self.mask.bool()[:num_tokens,:num_tokens]

        #apply mask to attn_scores
        attn_scores.masked_fill_(mask_bool,-torch.inf) #(batch_size,num_heads,num_tokens,num_tokens)

        #apply scaling and softmax 
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5,dim=-1)  #(batch_size,num_heads,num_tokens,num_tokens)

        #apply dropout
        attn_weights = self.dropout(attn_weights) #(batch_size,num_heads,num_tokens,num_tokens)

        #calculate context vec
        context_vec = attn_weights @ values #(batch_size,num_heads,num_tokens,head_dim)

        #bring num_heads and head_dim together so that we can change them to d_out
        context_vec = context_vec.transpose(1,2)
        #note : view demands matrix elements to be in contiguous memory but after transpose and permute elements may go to non cont memory 
        # so first apply contiguous which returns a contiguous copy of tensor in memory 
        context_vec = context_vec.contiguous().view(b,num_tokens,self.d_out)
        #final linear layer
        context_vec = self.out_proj(context_vec) #(b,num_tokens,d_out)
        return context_vec
    

    





        


