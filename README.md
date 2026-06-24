# Build Your Own ChatGPT From Scratch

Season of Code 2026
Mid-Term Submission | Mentee: Annu Kumari| Roll No: 25B0728 | Mentor:Nihar

---

## What is this project?

This project is part of the SoC program where we're trying to build a GPT-style language model completely from scratch - no using Hugging Face or pre-built transformer libraries. The goal is to understand what's actually happening inside ChatGPT, not just use it.

We're following the resources given by the mentors and also using Andrej Karpathy's "Let's build GPT" video as the main reference for the final implementation.

---

## Week-wise Progress

### Week 1 - Python + Data Science Basics
- Revised Python (numpy, pandas, matplotlib)
- Understood how to work with arrays and do basic data manipulation
- This week was more of a warmup honestly

### Week 2 — Neural Networks (Part 1)
- Watched StatQuest videos on neural networks (videos 74–80 from the playlist)
- Understood forward pass, loss functions, and backpropagation conceptually
- Implemented a small neural network manually using just numpy (no pytorch yet)
- Got confused by backprop math but eventually got it after watching twice

### Week 3 — Neural Networks (Part 2) + Intro to Transformers
- Finished remaining StatQuest videos (80–84)
- Started reading about what transformers are — watched 3b1b's attention video
- Learned what embeddings are and why we need them for text
- Tried to understand the "Attention is All You Need" paper - honestly understood maybe 40% of it

### Week 4 — PyTorch + Tokenization + Starting the Model
- Started learning PyTorch basics
- Implemented a character-level tokenizer from scratch (see `code/tokenizer.py`)
- Started writing the basic transformer skeleton (see `code/simple_model.py`)
- The self-attention part is still not fully clear to me, working on it

---

## Current Status

- [x] Python/numpy basics
- [x] Understand what neural networks are doing
- [x] Understand embeddings and tokenization
- [x] Built a basic character-level tokenizer
- [x] Understand (roughly) what self-attention does
- [ ] Full transformer block working
- [ ] Training loop (just started)
- [ ] Actual text generation
- [ ] Dataset preparation

Basically the code skeleton is there but the model doesn't actually train yet. I'm still figuring out how to properly implement multi-head attention and then connect everything into a training loop.

---

## Challenges I faced

1. **Backpropagation math** — the chain rule stuff took a while. I kept getting confused between the gradient of the loss w.r.t weights vs activations.

2. **Self-attention** — I understand the QKV concept now but implementing it without bugs is hard. The matrix dimensions kept mismatching.

3. **PyTorch vs numpy confusion** — kept forgetting that PyTorch tensors need `.item()` to get scalar values, and `.detach()` before converting to numpy.

4. **The paper** — "Attention is All You Need" is hard to read if you don't know the notation. Had to watch multiple videos before the paper made sense.

---

## Next Steps (Week 5 onwards)

- Finish implementing the full transformer block (with working multi-head attention)
- Write the training loop and test it on a tiny dataset (Shakespeare text probably)
- Get loss to actually decrease
- Eventually generate some text (even if it's garbage, at least the model runs)

---

## Resources I'm using

- SOC Resource Repo: https://github.com/Ahoy-Codey/ChatGPT-from-Scratch-Resource
- Andrej Karpathy's GPT tutorial: https://www.youtube.com/watch?v=kCc8FmEb1nY
- 3Blue1Brown attention series
- StatQuest Neural Networks playlist
