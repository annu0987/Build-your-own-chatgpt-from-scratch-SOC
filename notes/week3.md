# Week 3 Notes - Transformers and Attention

## The general idea

After going through neural network basics, week 3 moved into how transformers actually work. The main thing transformers are known for is the attention mechanism. RNNs (older models) processed text left to right, one word at a time. Transformers look at all the tokens at once.

---

## Self-Attention

The core question attention is answering: for each token in the sequence, how much should it "pay attention" to every other token?

The way this is computed:

1. Each token gets turned into three vectors: Query (Q), Key (K), Value (V)
2. The Q of one token is compared to the K of every other token (dot product)
3. This gives a score -- higher score means more relevant
4. Scores are scaled and passed through softmax to get weights (they sum to 1)
5. The final output for that token is a weighted sum of all the V vectors

In math:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
```

The `sqrt(d_k)` scaling is to stop the dot products from getting too large when the dimension is high. Without it softmax becomes very spiky apparently.

---

## Why this is useful

Say you have the sentence "the bank by the river was steep". When the model processes "bank", attention can look at "river" and figure out this isn't a financial bank. An RNN might lose that context if the sentence was longer.

---

## Multi-Head Attention

Instead of doing attention once, you do it multiple times with different Q, K, V projections. Each "head" can focus on different kinds of relationships. Then you concat the results and project them down.

This part is where I got confused. I understand why you'd want to look at multiple relationships, but the implementation -- keeping track of (batch, heads, seq_len, head_dim) -- gets hard to follow. I need to go through it with actual code more slowly.

---

## Transformer Block

A full transformer block is roughly:

```
x -> LayerNorm -> Self-Attention -> (add residual) -> LayerNorm -> Feed Forward -> (add residual) -> output
```

The residual connections (adding the input back to the output) help gradients flow during training. LayerNorm stabilizes activations.

The feed-forward part is just a small 2-layer MLP applied to each position independently. I'm not totally sure why this is needed on top of attention but it seems standard in every implementation.

---

## What confused me

Positional encodings. Attention by itself doesn't know the order of tokens. You could shuffle the sentence and get the same attention scores (because it's all dot products). So you add position information to the embeddings before passing them in.

The original paper used sine and cosine functions for this. Newer models learn positional embeddings instead. But I still don't fully get why the sine/cosine approach works or what exactly it's encoding. I'd like to revisit this.

---

## Small thing I noticed

The transformer doesn't process sequence positions in order. It's all matrix operations. This is why GPUs are good at this -- the whole sequence can be computed in parallel instead of step by step.
