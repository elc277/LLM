import torch

with open("input.txt", 'r', encoding='utf-8') as f:
    text=f.read()
    
#print("length in chars:",len(text))

chars=sorted(list(set(text)))
vocab_size=len(chars)

#tokenizer--replace later
stoi={ch:i for i,ch in enumerate(chars)}
itos={i:ch for i,ch in enumerate(chars)}
encode=lambda s:[stoi[c] for c in s]
decode=lambda l: ''.join([itos[i] for i in l])

data=torch.tensor(encode(text), dtype=torch.long)
#the data is encoded and stored into a torch tensor

#split data into training data and validation data -- 90%, 10%
n=int(0.9*len(data))
train_data=data[:n]
val_data=data[n:]