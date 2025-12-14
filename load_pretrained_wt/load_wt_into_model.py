import torch
import numpy as np


def assign(left,right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch, left:{left.shape},right:{right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt,params):
    #first lets assign for simplest ones 
    #pos_emb
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight,params['wpe'])
    #tok_emb
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight,params['wte'])

    #final layernorm
    gpt.final_norm.scale = assign(gpt.final_norm.scale,params['g'])
    gpt.final_norm.shift = assign(gpt.final_norm.scale,params['b'])

    #for out_head
    gpt.out_head.weight = assign(gpt.out_head.weight,params['wte'])
    
    #now finally for the transformer blocks 
    for b in range(len(params['blocks'])):
        #1)attention block
        #1)a) Query key and value matrix
        # 1)a)i) KEY, QUERY , VALUE - weights 
        W_query,W_key,W_value = np.split(params['blocks'][b]['attn']['c_attn']['w'],3,axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(gpt.trf_blocks[b].att.W_query.weight,W_query.T) 
        gpt.trf_blocks[b].att.W_key.weight = assign(gpt.trf_blocks[b].att.W_key.weight,W_key.T)    
        gpt.trf_blocks[b].att.W_value.weight = assign(gpt.trf_blocks[b].att.W_value.weight,W_value.T) 

        # 1)a)ii) KEY, QUERY , VALUE - bias 
        B_query,B_key,B_value = np.split(params['blocks'][b]['attn']['c_attn']['b'],3,axis=-1)
    
        gpt.trf_blocks[b].att.W_query.bias = assign(gpt.trf_blocks[b].att.W_query.bias,B_query) 
        gpt.trf_blocks[b].att.W_key.bias = assign(gpt.trf_blocks[b].att.W_key.bias,B_key)    
        gpt.trf_blocks[b].att.W_value.bias = assign(gpt.trf_blocks[b].att.W_value.bias,B_value) 

        #1) b) Out proj matrix
        gpt.trf_blocks[b].att.out_proj.weight = assign(gpt.trf_blocks[b].att.out_proj.weight,params['blocks'][b]['attn']['c_proj']['w'].T) 
        gpt.trf_blocks[b].att.out_proj.bias = assign(gpt.trf_blocks[b].att.out_proj.bias,params['blocks'][b]['attn']['c_proj']['b']) 


        #2) layer_norm 1
        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale, 
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift, 
            params["blocks"][b]["ln_1"]["b"])
        
        #3) Layer norm 2
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale, 
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift, 
            params["blocks"][b]["ln_2"]["b"])
        
        #4) mlp
        #4)i) fc layer weights and bias
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight, 
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias, 
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        
        #4)ii) proj layer weights and bias
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight, 
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias, 
            params["blocks"][b]["mlp"]["c_proj"]["b"])





