# Week 3 Notes — Transformers and Self-Attention

This week was the hardest so far. Took me a while to get the attention stuff.

## What's wrong with RNNs?

Before transformers, people used RNNs (Recurrent Neural Networks) for text. The problem was:
- They process tokens one by one, sequentially
- Hard to remember things from far back in the sequence (vanishing gradient problem)
- Can't be parallelized easily, so training is slow

Transformers fix all of this by processing all tokens simultaneously using **attention**.

## Self-Attention

The core idea: for each token in the sequence, self-attention lets it "look at" all other tokens and decide which ones are most relevant.

Example: in "The animal didn't cross the street because it was too tired"
- When processing "it", the model needs to figure out "it" refers to "animal"
- Self-attention lets "it" attend to "animal" and figure out the connection

### How it works (the math)

Each token produces three vectors:
- **Q (Query)**: "what am I looking for?"
- **K (Key)**: "what do I contain?"
- **V (Value)**: "what information do I provide?"

These come from multiplying the input embedding by three learned weight matrices (Wq, Wk, Wv).

Then:
1. Compute dot product of Q with all K's → this gives a score (how relevant is each token to me?)
2. Divide by sqrt(d_k) — this keeps gradients stable
3. Apply softmax → scores become probabilities (sum to 1)
4. Multiply probabilities by V → weighted sum of values

In matrix form:
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
```

```python
import torch
import torch.nn.functional as F

def simple_attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    weights = F.softmax(scores, dim=-1)
    output = torch.matmul(weights, V)
    return output
```

## Multi-Head Attention

Instead of doing attention once, we do it multiple times in parallel (each called a "head"), each with different weight matrices. Then we concatenate the results.

Why? Each head can focus on different kinds of relationships. One head might focus on syntax, another on semantics, etc.

## Transformer Block structure
