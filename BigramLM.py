import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

torch.manual_seed(2707)

# hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

# Read the text
with open("input.txt", 'r', encoding='utf-8') as f:
    text = f.read()

# Create and train a BPE tokenizer
def create_bpe_tokenizer(text, vocab_size=1000):
    # Initialize a tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(
        vocab_size=vocab_size, 
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    )
    
    # Pre-tokenize using whitespace
    tokenizer.pre_tokenizer = Whitespace()
    
    # Train the tokenizer
    tokenizer.train_from_iterator([text], trainer=trainer)
    
    return tokenizer

# Create the tokenizer with vocabulary size of 1000 (you can adjust this)
tokenizer = create_bpe_tokenizer(text, vocab_size=1000)
vocab_size = tokenizer.get_vocab_size()

print(f"Tokenizer vocabulary size: {vocab_size}")

# New encode/decode functions using the tokenizer
def encode(s):
    return tokenizer.encode(s).ids

def decode(l):
    return tokenizer.decode(l)

# Tokenize the entire text
data = torch.tensor(encode(text), dtype=torch.long)

# Split data into training and validation sets - 90%, 10% 
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    """
    Data loading
    """
    # Generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

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
        out[split] = losses.mean()
    model.train()
    return out

# Improved model - adding embedding dimension
class ImprovedBigramLM(nn.Module):
    def __init__(self, vocab_size, n_embd=64):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Get token embeddings
        tok_emb = self.token_embedding_table(idx)  # (B, T, n_embd)
        
        # Get position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=device)  # (T)
        pos_emb = self.position_embedding_table(pos)  # (T, n_embd)
        
        # Combine token and position embeddings
        x = tok_emb + pos_emb  # (B, T, n_embd)
        
        # Project back to vocabulary size
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        """
        Take the (B, T) tensor of indices and extend it by max_new_tokens.
        """
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop the context if it exceeds block_size
            idx_cond = idx[:, -block_size:] if idx.size(1) > block_size else idx
            
            # Get the predictions
            logits, _ = self(idx_cond)
            
            # Focus only on the last time step
            logits = logits[:, -1, :]  # (B, C)
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
            
        return idx

# Initialize the model
model = ImprovedBigramLM(vocab_size)
model = model.to(device)

# Create optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    # Sample a batch of data
    xb, yb = get_batch('train')
    
    # Evaluate the loss
    logits, loss = model(xb, yb)
    
    # Backpropagation
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_ids = model.generate(context, max_new_tokens=500)[0].tolist()
print(decode(generated_ids))