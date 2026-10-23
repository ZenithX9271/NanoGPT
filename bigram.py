"""
bigram.py — Phase 1: Bigram Language Model
-------------------------------------------
The simplest possible language model: predict the next character
purely based on the current character. No memory, no context.

This is our baseline before we build the full Transformer.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

# ── Reproducibility ──────────────────────────────────────────────────────────
torch.manual_seed(1337)

# ── Hyperparameters ──────────────────────────────────────────────────────────
batch_size  = 32      # how many independent sequences we process in parallel
block_size  = 8       # maximum context length for predictions
max_iters   = 3000    # training steps
eval_interval   = 300 # how often we evaluate the loss
eval_iters  = 200     # how many batches to average for the loss estimate
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {device}")

# ── Dataset ───────────────────────────────────────────────────────────────────
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Dataset: {len(text):,} characters")

# ── Vocabulary ────────────────────────────────────────────────────────────────
# Character-level tokeniser — every unique character is a token
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocab size: {vocab_size} unique characters")
print(f"Characters: {''.join(chars)}")

# Mappings: character <-> integer
stoi = {ch: i for i, ch in enumerate(chars)}   # string → int
itos = {i: ch for i, ch in enumerate(chars)}   # int → string

encode = lambda s: [stoi[c] for c in s]          # str  → list[int]
decode = lambda l: ''.join([itos[i] for i in l])  # list[int] → str

# ── Train / Val split ─────────────────────────────────────────────────────────
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))   # 90% train, 10% val
train_data = data[:n]
val_data   = data[n:]

# ── Data loader ───────────────────────────────────────────────────────────────
def get_batch(split):
    """
    Sample a random batch of (inputs, targets).
    
    For each sequence of length block_size, the target is the same sequence
    shifted by one position — i.e. for every position t, we want to predict
    data[t+1] given data[0..t].
    """
    data = train_data if split == 'train' else val_data
    # Random starting indices for each item in the batch
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size]     for i in ix])   # (B, T)
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])  # (B, T)
    x, y = x.to(device), y.to(device)
    return x, y

# ── Loss estimation ───────────────────────────────────────────────────────────
@torch.no_grad()
def estimate_loss(model):
    """Average loss over eval_iters batches for both splits."""
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

# ── Model ─────────────────────────────────────────────────────────────────────
class BigramLanguageModel(nn.Module):
    """
    Bigram model: an embedding table where row i contains the logits
    (un-normalised scores) for what character comes after character i.

    There is NO context — each prediction depends only on the current token.
    This is essentially a lookup table, not a real neural network.
    """

    def __init__(self, vocab_size):
        super().__init__()
        # Each token directly reads off the logits for the next token
        # from a lookup table of shape (vocab_size, vocab_size)
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        """
        idx:     (B, T) tensor of token indices
        targets: (B, T) tensor of target token indices (optional)

        Returns:
            logits: (B, T, C)  C = vocab_size
            loss:   scalar cross-entropy loss (or None if no targets)
        """
        # (B, T) → (B, T, C)
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            # F.cross_entropy expects (N, C) predictions and (N,) targets
            logits  = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        Autoregressively generate max_new_tokens new tokens.
        idx: (B, T) conditioning context.
        """
        for _ in range(max_new_tokens):
            # Get predictions
            logits, _ = self(idx)          # (B, T, C)
            
            # Focus only on the last time step (bigram: only last token matters):
            # Let's imagine we already generated "TH". Our idx is [[20, 8]] (indices for 'T' and 'H')
            # We don't care about the prediction after 'T' anymore (we already have 'H').
            # We only care about what comes after the current last character ('H').
            logits = logits[:, -1, :]      # (B, C)

            # Convert logits → probabilities
            probs = F.softmax(logits, dim=-1)   # (B, C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # Append to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


# ── Training ──────────────────────────────────────────────────────────────────
model = BigramLanguageModel(vocab_size)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("\nTraining Bigram model...")
for step in range(max_iters):

    # Evaluate loss periodically
    if step % eval_interval == 0:
        losses = estimate_loss(model)
        print(f"  step {step:4d}: train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

    # Sample a batch
    xb, yb = get_batch('train')

    # Forward + backward
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# ── Generation ────────────────────────────────────────────────────────────────
print("\n── Generated text (Bigram baseline) ──────────────────────────────────")
context = torch.zeros((1, 1), dtype=torch.long, device=device)  # start with \n
generated = model.generate(context, max_new_tokens=500)[0].tolist()
print(decode(generated))
print("──────────────────────────────────────────────────────────────────────")
