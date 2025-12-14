import torch 
from torch.utils.data import Dataset
import numpy as np

class GPTDataset(Dataset):
    def __init__(self,txt,tokenizer,context_length,stride):
        self.input_ids = []
        self.output_ids = []
        token_ids = tokenizer.encode(txt,allowed_special = {"<|endoftext|>"})  #shape(n,)
        print(np.array(token_ids).shape)
        print("Length of tokenized_text: ",len(token_ids))
        for i in range(0,len(token_ids)-context_length,stride):
            input_chunk = token_ids[i:i+context_length]
            output_chunk = token_ids[i+1:i+1+context_length]
            self.input_ids.append(torch.tensor(input_chunk))
            self.output_ids.append(torch.tensor(output_chunk))
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self,idx):
        return (self.input_ids[idx],self.output_ids[idx])
    
    