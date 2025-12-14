import torch 
from torch.utils.data import DataLoader
import tiktoken
from data.dataset import GPTDataset




with open('data/the-verdict.txt','r',encoding='utf8') as f:
    raw_text = f.read()

def create_dataloader(txt,context_length,stride,batch_size,shuffle = True,num_workers =0,drop_last = False):
    tokenizer = tiktoken.get_encoding('gpt2') 
    dataset = GPTDataset(txt,tokenizer,context_length,stride)
    print("Length of dataset is: ",len(dataset))
    dataloader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        num_workers = num_workers,
        drop_last = drop_last
    )
    return dataloader


dataloader = create_dataloader(raw_text,context_length=100,stride=100,batch_size=4,shuffle=False)
print("length of dataloader:",len(dataloader))
for key,value in dataloader:
    print(key.shape,value.shape)


