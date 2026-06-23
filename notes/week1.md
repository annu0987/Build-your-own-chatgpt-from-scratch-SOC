# Week 1 Notes — Intro + DS Tools

## What we did

Mostly revision of Python and learning numpy/pandas/matplotlib. This week was kind of a foundation week before getting into the actual ML stuff.

## Numpy stuff I learned

- Arrays are way faster than Python lists because they use contiguous memory
- Broadcasting — numpy can do operations on arrays of different shapes automatically
  - e.g. `np.array([1,2,3]) + 5` works fine, adds 5 to each element
- Useful functions: `np.dot`, `np.reshape`, `np.zeros`, `np.random.randn`

```python
import numpy as np

# dot product — used a lot in neural networks
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(np.dot(a, b))  # 32

# matrix multiply
A = np.random.randn(3, 4)
B = np.random.randn(4, 2)
print(np.matmul(A, B).shape)  # (3, 2)
```

## What are LLMs?

LLMs = Large Language Models. The basic idea is:

- You give the model some text (a "prompt")
- The model predicts what word/token should come next
- Then it takes that output, adds it to the input, and predicts the next one
- Repeat this many times → you get generated text

This is called **autoregressive generation**. The model is basically always doing "given everything before this, what comes next?"

## Next Token Prediction

This is the core task that GPT-style models are trained on.

Example:
- Input: `"The cat sat on the"`
- Model predicts: `"mat"` (or `"floor"` or `"roof"` — gives probabilities for all words)

The model outputs a probability distribution over the entire vocabulary. During training, we want the probability of the correct next word to be high. We push it in that direction using gradient descent.

Key insight: if you train a model on enough text with just this simple task (predict next word), it somehow learns grammar, facts, reasoning, etc. Nobody fully understands why this works as well as it does.

## Things I'm still confused about

- Why does predicting the next token lead to "understanding"? Like how does autocomplete become ChatGPT?
- How does the model decide what context from earlier in the sentence is relevant?

Will probably understand this better once we get to transformers.
