import torch
import torch.nn as nn
from torch.nn import functional as F
import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
import pickle
import time

torch.manual_seed(2707)

# Hyperparameters
batch_size = 64  # number of independent sequences processed in parallel
block_size = 256  # maximum context length for predictions
max_iters = 7000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.27
checkpoint_path = "gpt_model_checkpoint.pth"  # Path to save/load the model
tokenizer_path = "bpe_tokenizer.json"  # Path to save/load the tokenizer
vocab_size = 5000  # Size of the BPE vocabulary (adjust as needed)

# Load text
with open("input.txt", 'r', encoding='utf-8') as f:
    text = f.read()

# Tokenizer setup
def create_bpe_tokenizer(text, vocab_size=5000):
    if os.path.exists(tokenizer_path):
        print(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = Tokenizer.from_file(tokenizer_path)
    else:
        print(f"Training new BPE tokenizer with vocab_size={vocab_size}")
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"],
            min_frequency=2
        )
        
        # Pre-tokenize using whitespace
        tokenizer.pre_tokenizer = Whitespace()
        
        # Train the tokenizer
        start_time = time.time()
        tokenizer.train_from_iterator([text], trainer=trainer)
        print(f"Tokenizer training completed in {time.time() - start_time:.2f} seconds")
        
        # Add post-processing to handle special tokens
        tokenizer.post_processor = TemplateProcessing(
            single="[BOS] $A [EOS]",
            special_tokens=[
                ("[BOS]", tokenizer.token_to_id("[BOS]")),
                ("[EOS]", tokenizer.token_to_id("[EOS]"))
            ]
        )
        
        # Save the tokenizer
        tokenizer.save(tokenizer_path)
        print(f"Tokenizer saved to {tokenizer_path}")
    
    return tokenizer

# Create the tokenizer
tokenizer = create_bpe_tokenizer(text, vocab_size=vocab_size)
vocab_size = tokenizer.get_vocab_size()
print(f"Vocabulary size: {vocab_size}")

# Create encoding and decoding functions
def encode(s):
    return tokenizer.encode(s).ids

def decode(ids):
    return tokenizer.decode(ids)

# Create dataset
print("Tokenizing entire dataset...")
start_time = time.time()
data = torch.tensor(encode(text), dtype=torch.long, device=device)
print(f"Dataset tokenized in {time.time() - start_time:.2f} seconds")
print(f"Sequence length after tokenization: {len(data)} tokens")

# Split data into training and validation sets
n = int(0.85 * len(data))
train_data = data[:n]
val_data = data[n:]

# Data loading function
def get_batch(split):
    """
    Generate a small batch of data of inputs x and targets y
    """
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

# Loss estimation function
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

# Self-attention head
class Head(nn.Module):
    """One head of self-attention"""
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        
        # Compute attention scores (affinities)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        # Perform weighted aggregation of values
        v = self.value(x)
        out = wei @ v
        return out

# Multi-head attention
class MultiheadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

# Feed-forward network
class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity"""
    
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),  # Changed from ReLU to GELU for better performance
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
        
    def forward(self, x):
        return self.net(x)

# Transformer block
class Block(nn.Module):
    """Transformer block: communication followed by computation"""
    
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiheadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        # Using pre-norm formulation (better for training stability)
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# GPT Language Model
class GPTLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Get token and position embeddings
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        
        # Apply transformer blocks
        x = self.blocks(x)
        x = self.ln_f(x)
        
        # Get logits
        logits = self.lm_head(x)
        
        # Calculate loss if targets are provided
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate text by sampling tokens from the model's distribution
        Args:
            idx: context tokens
            max_new_tokens: number of tokens to generate
            temperature: controls randomness (lower = more deterministic)
            top_k: if set, sample from top k most likely tokens
        """
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Crop the context if needed
                idx_cond = idx[:, -block_size:] if idx.size(1) > block_size else idx
                
                # Get predictions
                logits, _ = self(idx_cond)
                
                # Focus on the last token
                logits = logits[:, -1, :] / temperature
                
                # Optional top-k sampling
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')
                
                # Apply softmax to get probabilities
                probs = F.softmax(logits, dim=-1)
                
                # Sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)
                
                # Append to the sequence
                idx = torch.cat((idx, idx_next), dim=1)
        
        return idx

# Create model
print(f"Creating GPT model with vocab_size={vocab_size}")
model = GPTLM(vocab_size)
model = model.to(device)

# Create optimizer with weight decay
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)

# Learning rate scheduler with warmup
def get_lr_scheduler(optimizer):
    def lr_lambda(step):
        # Warmup for 200 steps
        warmup_steps = 200
        if step < warmup_steps:
            return step / warmup_steps
        # Linear decay after warmup
        return max(0.1, 1.0 - (step - warmup_steps) / (max_iters - warmup_steps))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

scheduler = get_lr_scheduler(optimizer)

# Training or loading the model
if os.path.exists(checkpoint_path):
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("Model loaded successfully.")
else:
    print("No checkpoint found. Training from scratch...")
    best_val_loss = float('inf')
    
    for iter in range(max_iters):
        # Evaluate
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {scheduler.get_last_lr()[0]:.6f}")
            
            # Save best model
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'iter': iter,
                    'val_loss': best_val_loss
                }, checkpoint_path)
                print(f"New best model saved at iter {iter} with val loss {best_val_loss:.4f}")
        
        # Sample a batch
        xb, yb = get_batch('train')
        
        # Forward pass
        logits, loss = model(xb, yb)
        
        # Skip step if loss is NaN
        if torch.isnan(loss) or torch.isinf(loss):
            print("NaN detected in loss, skipping step.")
            continue
        
        # Backward pass and optimize
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()
        scheduler.step()
    
    print("Training complete.")

# Generate text
print("\nGenerating text...")
context = torch.tensor([[tokenizer.token_to_id("[BOS]")]], dtype=torch.long, device=device)
generated_ids = model.generate(context, max_new_tokens=5000, temperature=0.8, top_k=40)[0].tolist()
generated_text = decode(generated_ids)

print("\nGenerated text sample (first 500 chars):")
print(generated_text[:500])

# Save generated text
with open("output.txt", "w", encoding="utf-8") as f:
    f.write(generated_text)

print("Full text saved in output.txt")