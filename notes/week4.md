# Week 4 Notes — Training Pipeline

This week I started actually writing code in PyTorch and trying to put together a training loop.

## PyTorch basics I learned

```python
import torch
import torch.nn as nn

# Tensors are like numpy arrays but with GPU support + autograd
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# Operations build a computation graph
y = x ** 2 + 3 * x
loss = y.sum()

# Backprop — computes gradients automatically
loss.backward()
print(x.grad)  # gradients of loss w.r.t x
```

## How training works for a language model

1. **Get a batch of text** — take a chunk of the training data
2. **Create input-output pairs** — input is tokens 0..n-1, target is tokens 1..n (shifted by 1)
3. **Forward pass** — feed input through model, get predicted probabilities for each position
4. **Compute loss** — compare predictions to targets using cross-entropy loss
5. **Backward pass** — compute gradients
6. **Update weights** — optimizer (usually Adam) adjusts weights to reduce loss
7. **Repeat**

```python
# Very simplified training step
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(num_steps):
    xb, yb = get_batch(data)           # get input and target
    logits, loss = model(xb, yb)       # forward pass
    optimizer.zero_grad()              # clear old gradients
    loss.backward()                    # backprop
    optimizer.step()                   # update weights
    
    if step % 100 == 0:
        print(f"step {step}, loss: {loss.item():.4f}")
```

## Cross-entropy loss

For language models, the loss at each position is:
