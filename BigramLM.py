import torch
import torch.nn as nn
from torch.nn import functional as F 
torch.manual_seed(2707)

#hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200


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

def get_batch(split):
    """
    data loading
    """
    #generate a small batch of data of inputs x and targets y
    data=train_data if split == 'train' else val_data
    ix = torch.randint(len(data)-block_size,(batch_size,))
    x=torch.stack([data[i:i+block_size]for i in ix])
    y=torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x,y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split]=losses.mean()
    model.train()
    return out

#BigramLanguageModel
class BigramLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table=nn.Embedding(vocab_size, vocab_size)
        
    def forward(self, idx, targets=None):
        logits=self.token_embedding_table(idx) #scores for the next char in the sequence
        #aranging in a (Batch, Time, Channel) tensor
        if targets is None:
            loss=None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) #strech out the tensor into a 2 dimensional array
            targets = targets.view(-1) #stretch out into a 1D array
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        """
        Take the (Batch*Time) tensor and extend it.
        It generates the extensions in the time dimension. 
        """
        #idx is (Batch, Time) array of indices
        for _ in range(max_new_tokens):
            logits, loss = self(idx) #get the predicitons
            logits = logits[:, -1, :] #focus only on the last step
            prob = F.softmax(logits, dim=-1) # apply softmax for probabilities
            idx_next = torch.multinomial(prob, num_samples=1) #sample from the distribution
            idx = torch.cat((idx, idx_next), dim=1) #append sampled index to the running sequence
        return idx
    
model=BigramLM(vocab_size)
m=model.to(device)
#print(logits.shape)
#print(loss)

optimizer=torch.optim.AdamW(model.parameters(), lr=learning_rate)

batch_size = 32
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    xb,yb = get_batch('train')
    
    logits, loss=model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context=torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))