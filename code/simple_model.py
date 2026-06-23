# simple_model.py
# A basic GPT-style transformer skeleton
# This is NOT fully working yet — self-attention is implemented
# but the full training loop has a bug I'm still debugging
# Written following Karpathy's tutorial + my own understanding from week 3-4

import torch
import torch.nn as nn
import torch.nn.functional as F

# hyperparameters — keeping everything small so it actually runs on CPU
BLOCK_SIZE = 32      # how many tokens the model looks at at once (context length)
EMBED_DIM = 64       # size of each token's embedding vector
NUM_HEADS = 4        # number of attention heads
NUM_LAYERS = 2       # how many transformer blocks to stack
DROPOUT = 0.1        # randomly zero out some activations during training (prevents overfitting)


class SelfAttentionHead(nn.Module):
    """
    One head of self-attention.
    
    Each head learns to focus on different kinds of relationships
    between tokens. When we use multiple heads (MultiHeadAttention below),
    the model can capture many different patterns at once.
    """
    
    def __init__(self, head_size):
        super().__init__()
        # three linear layers to produce Q, K, V from the input
        # bias=False because the original paper doesn't use bias here
        self.query = nn.Linear(EMBED_DIM, head_size, bias=False)
        self.key   = nn.Linear(EMBED_DIM, head_size, bias=False)
        self.value = nn.Linear(EMBED_DIM, head_size, bias=False)
        
        # causal mask — lower triangular matrix
        # position i can only attend to positions 0..i (not the future)
        self.register_buffer(
            'mask',
            torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
        )
        
        self.dropout = nn.Dropout(DROPOUT)
    
    def forward(self, x):
        B, T, C = x.shape  # B=batch, T=sequence length, C=embedding dim
        
        q = self.query(x)  # (B, T, head_size)
        k = self.key(x)    # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)
        
        head_size = q.shape[-1]
        
        # attention scores — dot product of queries and keys
        # scaled by sqrt(head_size) to keep values from getting too large
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_size ** 0.5)
        # scores shape: (B, T, T)
        
        # apply causal mask: future positions become -inf
        # after softmax, -inf → 0 (those positions are ignored)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        
        # softmax to get attention weights (probabilities)
        weights = F.softmax(scores, dim=-1)  # (B, T, T)
        weights = self.dropout(weights)
        
        # weighted sum of values
        out = torch.matmul(weights, v)  # (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """
    Run multiple attention heads in parallel, then concatenate their outputs.
    
    The idea is each head can "look for" different things.
    One head might focus on the subject of a sentence, another on the verb, etc.
    """
    
    def __init__(self, num_heads):
        super().__init__()
        head_size = EMBED_DIM // num_heads  # split embedding across heads
        
        self.heads = nn.ModuleList([
            SelfAttentionHead(head_size) for _ in range(num_heads)
        ])
        
        # project concatenated head outputs back to embed_dim
        self.proj = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.dropout = nn.Dropout(DROPOUT)
    
    def forward(self, x):
        # run all heads and concatenate along last dimension
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    """
    Simple 2-layer MLP applied to each token independently.
    
    After attention (which mixes information across tokens),
    this processes each token's representation on its own.
    The original paper uses hidden dim = 4 * embed_dim.
    """
    
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(EMBED_DIM, 4 * EMBED_DIM),
            nn.ReLU(),
            nn.Linear(4 * EMBED_DIM, EMBED_DIM),
            nn.Dropout(DROPOUT),
        )
    
    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    One transformer block = self-attention + feed-forward + residual connections + layer norm.
    
    Residual connections (the x + ... parts) help gradients flow during training.
    Layer norm normalizes each token's embedding — helps training stability.
    """
    
    def __init__(self):
        super().__init__()
        self.attention = MultiHeadAttention(NUM_HEADS)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(EMBED_DIM)
        self.ln2 = nn.LayerNorm(EMBED_DIM)
    
    def forward(self, x):
        # apply layer norm BEFORE attention (this is "pre-norm" — slightly different
        # from the original paper which does post-norm, but works better in practice)
        x = x + self.attention(self.ln1(x))  # residual connection
        x = x + self.ffwd(self.ln2(x))       # residual connection
        return x


class SimpleGPT(nn.Module):
    """
    The full model.
    
    Stack of transformer blocks with token + position embeddings at the start
    and a linear projection to vocabulary at the end.
    
    TODO: training loop is buggy, need to debug forward pass
    """
    
    def __init__(self, vocab_size):
        super().__init__()
        
        # token embedding: each token ID → a vector
        self.token_embedding = nn.Embedding(vocab_size, EMBED_DIM)
        
        # position embedding: each position (0, 1, ..., BLOCK_SIZE-1) → a vector
        # the model needs to know WHERE in the sequence each token is
        self.position_embedding = nn.Embedding(BLOCK_SIZE, EMBED_DIM)
        
        # stack of transformer blocks
        self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(NUM_LAYERS)])
        
        # final layer norm
        self.ln_final = nn.LayerNorm(EMBED_DIM)
        
        # project from embed_dim to vocab_size (these are the raw "logits")
        self.lm_head = nn.Linear(EMBED_DIM, vocab_size)
    
    def forward(self, idx, targets=None):
        """
        idx: token IDs, shape (B, T)
        targets: next token IDs for computing loss, shape (B, T)
                 if None, just return logits (for generation)
        """
        B, T = idx.shape
        
        # get token embeddings
        tok_emb = self.token_embedding(idx)         # (B, T, EMBED_DIM)
        
        # get position embeddings
        positions = torch.arange(T, device=idx.device)  # [0, 1, 2, ..., T-1]
        pos_emb = self.position_embedding(positions)     # (T, EMBED_DIM)
        
        # add them together — this is the input to the transformer
        x = tok_emb + pos_emb  # broadcasting handles the batch dimension
        
        # pass through transformer blocks
        x = self.blocks(x)     # (B, T, EMBED_DIM)
        x = self.ln_final(x)   # (B, T, EMBED_DIM)
        
        # project to vocabulary
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        # compute loss if targets are provided
        loss = None
        if targets is not None:
            # reshape for cross_entropy: (B*T, vocab_size) vs (B*T,)
            B, T, V = logits.shape
            logits_flat = logits.view(B * T, V)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        """
        Generate new tokens autoregressively.
        Keep feeding the model's output back in as input.
        
        NOTE: this function is written but untested — model doesn't train yet.
        """
        for _ in range(max_new_tokens):
            # crop context to BLOCK_SIZE (model can't handle longer sequences)
            idx_context = idx[:, -BLOCK_SIZE:]
            
            # get predictions
            logits, _ = self(idx_context)
            
            # only care about the last position's prediction
            last_logits = logits[:, -1, :]  # (B, vocab_size)
            
            # convert logits to probabilities
            probs = F.softmax(last_logits, dim=-1)
            
            # sample from the distribution (instead of always picking the top one)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            
            # append to sequence and continue
            idx = torch.cat([idx, next_token], dim=1)
        
        return idx


# --- quick sanity check ---
if __name__ == "__main__":
    # fake vocab size for testing
    vocab_size = 65
    
    model = SimpleGPT(vocab_size)
    
    # count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # test forward pass with random data
    batch_size = 4
    x = torch.randint(0, vocab_size, (batch_size, BLOCK_SIZE))
    y = torch.randint(0, vocab_size, (batch_size, BLOCK_SIZE))
    
    logits, loss = model(x, y)
    
    print(f"Logits shape: {logits.shape}")   # should be (4, 32, 65)
    print(f"Loss: {loss.item():.4f}")        # should be around ln(65) ≈ 4.17 initially
