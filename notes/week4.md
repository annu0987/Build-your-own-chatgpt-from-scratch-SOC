# Week 4 Notes - Training Flow and Model Pipeline

## What this week covered

Continued with the StatQuest playlist (videos 86-91). These videos went into how you actually train neural networks -- loss functions, optimizers, training loops. Also started thinking about how to put it all together into something that runs.

I didn't implement a full training run yet. That's supposed to come in Week 5 and 6 when we get to the actual GPT code.

---

## The Training Loop

The basic structure of training a neural network:

```
for each batch:
    1. forward pass (input -> model -> prediction)
    2. compute loss (compare prediction to target)
    3. backward pass (compute gradients)
    4. update weights (optimizer step)
    5. zero the gradients (so they don't accumulate)
```

Step 5 is easy to forget. PyTorch accumulates gradients by default, so you need to call `optimizer.zero_grad()` at the start of each iteration or the gradients stack up from previous batches.

---

## Loss Function

For language modeling the loss is cross-entropy. The model outputs a probability distribution over the vocabulary (which token comes next). The loss penalizes the model if it assigned low probability to the correct next token.

So if the correct next token is "cat" and the model said there's a 2% chance of "cat", that's a high loss. If it said 80%, low loss.

---

## Optimizer

Adam is the default for most things. It adapts the learning rate for each parameter based on how gradients have been behaving. SGD also works but Adam usually converges faster in practice.

The learning rate is a hyperparameter. Too high and training is unstable. Too low and it takes forever. I haven't experimented with this much yet.

---

## Next Token Prediction

The actual task of a language model is: given tokens so far, predict the next one. During training, you take a chunk of text, shift it by one, and train the model to predict each token from the ones before it.

```
input:  "the cat sat on"
target: "cat sat on the"
```

Every position in the sequence is a training example. This is why language modeling is cheap in data terms -- any text works.

---

## What I explored

- Understood how data is prepared (chunked into fixed-length sequences)
- Looked at how a simple training loop is structured in PyTorch
- Wrote a basic tokenizer (character-level, see `code/tokenizer.py`)
- Wrote a skeleton model that outlines the pieces (see `code/simple_model.py`)

## What I didn't implement

- Actual training -- the code doesn't run a full training loop yet
- Saving and loading model checkpoints
- Any evaluation / generating text from the model

The final project (week 6) is supposed to be a full GPT-like model trained on a custom dataset. I still have a ways to go before that.
