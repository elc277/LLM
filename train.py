import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(2707)

batch_size = 64 # number of independent sequences processed in parallel
block_size = 256 # maximum context length for predictions
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

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

data=torch.tensor(encode(text), dtype=torch.long, device=device)
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

class Head(nn.Module):
    """
    One head of self attention
    """
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size))) #Add the lower triangular matrix
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        #Compute attention scores -- affinities
        
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # mask the values so the future tokens dont comunicate with the past tokens
        wei = F.softmax(wei, dim=-1) #apply softmax
        wei = self.dropout(wei)
        v = self.value(x) # aggregate the values
        out = wei @ v
        return out

class MultiheadAttention(nn.Module):
    """
    multiple heads of self-attention in parallel
    """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """
    a linear layer followed by a non-linearity
    """
    
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4* n_embd, n_embd),
            nn.Dropout(dropout),
        )
        
    def forward(self,x):
        return self.net(x)

class Block(nn.Module):
    """
    Transformer Block
    Communication followed by computation 
    """
    
    def __init__(self, n_embd, n_head):
        #n_embd -- embedding dimension 
        #n_head -- the number of heads we want
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiheadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        #residual connections
        return x


class GPTLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) 
        self.lm_head = nn.Linear(n_embd, vocab_size)
        #tokens directly read off logits for the next tokens from a lookup table
        
    def forward(self, idx, targets=None):
        B,T = idx.shape
        
        tok_emb = self.token_embedding_table(idx) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=device, dtype=torch.long))
        x = tok_emb + pos_emb
        x = self.blocks(x) #use one head of self attention
        x = self.ln_f(x)
        logits = self.lm_head(x)
        #aranging in a (Batch, Time, Channel) tensor
        if targets is None:
            loss=None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) #strech out the tensor into a 2 dimensional array
            targets = targets.view(B*T) #stretch out into a 1D array
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        """
        Take the (Batch*Time) tensor and extend it.
        It generates the extensions in the time dimension. 
        """
        #idx is (Batch, Time) array of indices
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] #keep the context to a maximum of block_size
            logits, loss = self(idx_cond) #get the predicitons
            logits = logits[:, -1, :] #focus only on the last step
            prob = F.softmax(logits, dim=-1) # apply softmax for probabilities
            idx_next = torch.multinomial(prob, num_samples=1) #sample from the distribution
            idx = torch.cat((idx, idx_next), dim=1) #append sampled index to the running sequence
        return idx

model=GPTLM(vocab_size)
m=model.to(device)
#print(logits.shape)
#print(loss)

optimizer=torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters -1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    xb,yb = get_batch('train')
    
    logits, loss=model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    if torch.isnan(loss) or torch.isinf(loss):
        print("NaN detected in loss, skipping step.")
        continue
    loss.backward()
    optimizer.step()
    print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1e6} MB")

context=torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))