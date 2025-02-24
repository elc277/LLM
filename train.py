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

block_size = 8
train_data[:block_size+1]

x=train_data[:block_size] #inputs to the transformer -- first block_size chars 
y=train_data[1:block_size+1] #targets for each position in the input -- next block_size chars

for t in range(block_size):
    context=x[:t+1] # all the chars in text up to t(including t)
    target=y[t] # the t-th char
    #print(f"when input is {context} the target is: {target} ")

torch.manual_seed(1337)#change this in the future
batch_size = 4 # number of independent sequences processed in parallel
block_size = 8 # maximum context length for predictions

def get_batch(split):
    #generate a small batch of data of inputs x and targets y
    data=train_data if split == 'train' else val_data
    ix = torch.randint(len(data)-block_size,(batch_size,))
    x=torch.stack([data[i:i+block_size]for i in ix])
    y=torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x,y

xb,yb = get_batch('train')


