# Week 2 Notes - Neural Networks, Tokenization, Embeddings

## Neural Networks (beginning)

Started the StatQuest neural network playlist from video 74. Josh Starmer explains things simply which helped. The core idea is:

- You have inputs
- They pass through layers with weights
- Each layer applies some transformation
- At the end you get a prediction
- You compare that to the actual answer and get a loss
- You adjust weights to reduce loss

Gradient descent is the process of adjusting weights. The gradient tells you which direction to move in to reduce the loss. You move a small amount (learning rate) each step.

Backpropagation is how you compute those gradients. The chain rule from calculus lets you trace back through each layer and figure out how much each weight contributed to the error. The math makes sense on paper but I still find it hard to track mentally when there are many layers.

---

## Tokenization

Before text goes into a model it needs to be turned into numbers. Tokenization is that step.

The simplest version: split the text into words and assign each word an integer.

```
"the cat sat" -> [4, 12, 7]
```

But real tokenizers don't just split on spaces. They split into subword pieces. So "playing" might become ["play", "##ing"] or something like that. This helps with rare words.

Example with character-level tokenization:
```
text = "hello"
chars = sorted(set(text))       # ['e', 'h', 'l', 'o']
stoi = {c: i for i, c in enumerate(chars)}
encoded = [stoi[c] for c in text]
# encoded = [1, 0, 2, 2, 3]
```

The vocab size matters. Bigger vocab = fewer tokens per sentence but more memory for the embedding table.

---

## Embeddings

Once you have token IDs you still need to represent them as vectors, because the model works with vectors not integers. An embedding table is just a lookup: token ID 4 -> some vector of floats.

```
vocab_size = 100
embed_dim = 16
table = np.random.randn(vocab_size, embed_dim)

token_id = 4
embedding = table[token_id]   # shape: (16,)
```

The embedding starts random. During training it gets adjusted so tokens with similar meaning end up with similar vectors. At least that's the idea. I haven't actually seen it happen yet because I haven't trained anything.

One thing that surprised me: the embedding dimension is something you choose. It's a hyperparameter. 16, 64, 512 -- the bigger it is the more information can be packed in but also more computation.

---

## What I'm still fuzzy on

Why do embeddings "learn" to be meaningful? I get that the loss propagates back through them, but it's not obvious to me why similar words end up close together. I think it has to do with context but I need to look into this more.
