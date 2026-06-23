# tokenizer.py
# Character-level tokenizer — converts text to integers and back
# This is the simplest kind of tokenizer, not what GPT actually uses
# but good enough for learning the concept

class CharTokenizer:
    def __init__(self, text):
        # get all unique characters from the training text
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        
        # build lookup tables
        self.char_to_int = {ch: i for i, ch in enumerate(self.chars)}
        self.int_to_char = {i: ch for i, ch in enumerate(self.chars)}
        
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Characters: {''.join(self.chars)}")
    
    def encode(self, text):
        """Convert a string to a list of integers."""
        result = []
        for ch in text:
            if ch not in self.char_to_int:
                # skip characters not in vocab
                # in real tokenizers you'd handle this differently
                continue
            result.append(self.char_to_int[ch])
        return result
    
    def decode(self, tokens):
        """Convert a list of integers back to a string."""
        return ''.join([self.int_to_char[i] for i in tokens])


# --- test it out ---
if __name__ == "__main__":
    # small example text
    sample_text = """To be, or not to be, that is the question.
Whether tis nobler in the mind to suffer
the slings and arrows of outrageous fortune."""
    
    tokenizer = CharTokenizer(sample_text)
    
    test_sentence = "To be, or not"
    encoded = tokenizer.encode(test_sentence)
    decoded = tokenizer.decode(encoded)
    
    print(f"\nOriginal : {test_sentence}")
    print(f"Encoded  : {encoded}")
    print(f"Decoded  : {decoded}")
    print(f"Round-trip OK: {test_sentence == decoded}")
    
    # show a few mappings
    print("\nSome token mappings:")
    for ch, idx in list(tokenizer.char_to_int.items())[:10]:
        print(f"  '{ch}' -> {idx}")
