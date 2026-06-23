# Week 2 Notes — Tokenization and Embeddings

## Tokenization

Before a model can process text, we have to convert it into numbers. That's what tokenization does.

There are different ways to tokenize:

### Character-level tokenization
- Split text into individual characters
- Vocabulary is small (like 70-100 characters)
- Simple to implement
- Problem: sequences get very long, model has to learn to combine characters into words

### Word-level tokenization
- Split on spaces basically
- Vocabulary gets huge (100k+ words)
- Doesn't handle unknown words well ("ChatGPT" wouldn't be in a vocabulary from 2015)

### Subword tokenization (what GPT actually uses)
- Something in between — breaks words into common subword pieces
- e.g. "unbelievable" → ["un", "believ", "able"]
- GPT uses BPE (Byte Pair Encoding) — merges the most common pairs of characters/tokens repeatedly
- We're not implementing BPE from scratch right now, just doing character-level for simplicity

## My character tokenizer (simplified version)

```python
text = "hello world"
chars = sorted(set(text))          # unique characters
char_to_int = {c: i for i, c in enumerate(chars)}
int_to_char = {i: c for i, c in enumerate(chars)}

# encode
encoded = [char_to_int[c] for c in text]
print(encoded)

# decode
decoded = ''.join([int_to_char[i] for i in encoded])
print(decoded)  # should give back "hello world"
```

## Embeddings

After tokenizing, we have integer IDs. But these integers don't carry any meaning by themselves — the number 42 doesn't mean anything more than the number 7.

So we use **embeddings**: each token ID gets mapped to a dense vector of real numbers (e.g. 64 or 128 dimensions).

These vectors are learned during training. Similar words end up with similar vectors.

Example: after training on enough text, the vectors for "king" and "queen" will be closer together than "king" and "banana".

In PyTorch:
```python
import torch
import torch.nn as nn

vocab_size = 65    # number of unique tokens
embed_dim = 32     # how many dimensions per token

embedding = nn.Embedding(vocab_size, embed_dim)

# convert a token ID to a vector
token_id = torch.tensor([5])
vector = embedding(token_id)
print(vector.shape)  # (1, 32)
```

## Positional Encoding

One problem: embeddings don't tell the model where in the sequence a token appears. "cat sat on mat" and "mat on sat cat" have the same tokens, just different order.

To fix this, we add **positional encodings** — vectors that encode position — to the token embeddings.

The original paper used sine/cosine functions for this. In simpler implementations, we just learn a separate embedding for each position (works fine for shorter sequences).

## Summary

text → tokenize → integer IDs → embedding lookup → vectors → feed to model

This pipeline is the entry point for the entire transformer.
