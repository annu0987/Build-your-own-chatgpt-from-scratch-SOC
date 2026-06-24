# tokenizer.py
# character level tokenizer - very basic
# following along from the week 2 material

text = open("input.txt", "r").read() if __import__("os").path.exists("input.txt") else "hello world this is a test"

# get all unique characters
chars = sorted(set(text))
vocab_size = len(chars)

print(f"total characters in text: {len(text)}")
print(f"vocab size: {vocab_size}")
print(f"characters: {''.join(chars)}")

# maps: character -> integer and back
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# encode and decode functions
def encode(s):
    return [stoi[c] for c in s]

def decode(ids):
    return "".join([itos[i] for i in ids])


# quick test
sample = "hello"
# filter only chars that exist in vocab
sample = "".join([c for c in sample if c in stoi])

encoded = encode(sample)
decoded = decode(encoded)

print(f"\nsample: '{sample}'")
print(f"encoded: {encoded}")
print(f"decoded back: '{decoded}'")

# I was expecting this to look more complicated
# but character-level is actually pretty simple
# the real tokenizers (BPE etc) are harder

# not sure how to handle unknown characters properly
# right now it would just crash if a char isn't in vocab
