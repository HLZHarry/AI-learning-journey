# Day 3 - AI Track: Tokenization

**Date**: Day 3 Progress  
**Focus**: Understanding Tokenization in LLMs  
**Resource**: https://huggingface.co/learn/llm-course/chapter2/4

---

## What I Learned

### 1. What is Tokenization?

**Definition**: Breaking text into smaller units (tokens) that models can process

**Why Needed**:
- Models can't understand raw text
- Need numerical representation
- Balance between vocabulary size and granularity

---

## 2. Tokenization Methods

### **BPE (Byte-Pair Encoding)** - Used by GPT-2

**How it works**:
```
1. Start with characters
2. Merge most frequent pairs iteratively
3. Build subword vocabulary

Example:
"tokenization" → ["token", "ization"]
```

**Properties**:
- Handles unknown words well
- Smaller vocabulary
- Can represent any text

---

### **WordPiece** - Used by BERT

**How it works**:
```
Similar to BPE but:
- Uses ## prefix for subwords
- Different merging algorithm

Example:
"playing" → ["play", "##ing"]
```

**Properties**:
- Good for morphologically rich languages
- Uses ## to mark subwords
- Efficient vocabulary usage

---

## 3. Experiments Conducted

### Experiment 1: Compare Tokenizers

**Code**:
```python
from transformers import GPT2Tokenizer, BertTokenizer

gpt2 = GPT2Tokenizer.from_pretrained('gpt2')
bert = BertTokenizer.from_pretrained('bert-base-uncased')

text = "Hello, how are you?"
gpt2_tokens = gpt2.tokenize(text)
bert_tokens = bert.tokenize(text)
```

**Test Cases**:
1. Simple English: "Hello, how are you?"
2. Chinese: "人工智能真厉害！"
3. Code: "print('Hello World')"
4. Long word: "supercalifragilisticexpialidocious"
5. Sentence: "The quick brown fox jumps over the lazy dog."

**Key Finding**: Different tokenizers split text differently!

---

### Experiment 2: Special Tokens

**BERT Special Tokens**:
```
[CLS]: Classification token (start of sequence)
[SEP]: Separator token (end of sequence)
[PAD]: Padding token (fill to max length)
[UNK]: Unknown token (out-of-vocabulary words)
```

**Usage Example**:
```python
encoded = tokenizer.encode_plus(
    "Hello world",
    add_special_tokens=True,    # Add [CLS], [SEP]
    max_length=10,              # Set max length
    padding='max_length',       # Pad to max_length
    return_tensors='pt'         # Return PyTorch tensors
)
```

**Output Structure**:
```
input_ids:      [CLS] Hello world [SEP] [PAD] [PAD] ...
attention_mask: 1     1     1     1     0     0     ...
```

---

### Experiment 3: Token Limits

**Context Window**:
- GPT-2: 1024 tokens max
- BERT: 512 tokens max
- Modern models: 2048-100k+ tokens

**Problem**: Text longer than limit

**Solutions**:
```python
# Solution 1: Truncate
tokens = tokens[:1024]

# Solution 2: Chunk
chunks = [tokens[i:i+1024] for i in range(0, len(tokens), 1024)]

# Solution 3: Sliding window
# Process with overlapping windows
```

---

## 4. Key Concepts

### Token vs Word

```
Word:  "playing"
Tokens: ["play", "##ing"]  (WordPiece)
        ["play", "ing"]     (BPE)

Why split?
- Smaller vocabulary
- Handle unknown words
- Capture morphology
```

---

### Vocabulary Size

**Trade-off**:
```
Small vocab (e.g., 1000):
✓ Less memory
✗ More tokens per text
✗ Longer sequences

Large vocab (e.g., 50000):
✓ Fewer tokens per text
✓ Shorter sequences
✗ More memory
✗ More parameters
```

**Common sizes**:
- GPT-2: ~50k tokens
- BERT: ~30k tokens

---

### Encoding vs Decoding

**Encoding** (Text → IDs):
```python
text = "Hello world"
ids = tokenizer.encode(text)
# [15496, 995]
```

**Decoding** (IDs → Text):
```python
ids = [15496, 995]
text = tokenizer.decode(ids)
# "Hello world"
```

**Important**: Tokenization is **reversible**!

---

## 5. Practical Insights

### Why Different Tokenizers?

**GPT-2 (BPE)**:
- Good for generation
- Handles any text
- No [CLS]/[SEP] tokens

**BERT (WordPiece)**:
- Good for understanding
- Uses special tokens
- Case-sensitive option

---

### Handling Chinese Text

**Observation**:
```
Chinese: "人工智能真厉害！"

GPT-2: Many tokens (character-level)
BERT: Fewer tokens (trained on multilingual data)
```

**Lesson**: Tokenizer performance varies by language!

---

### Code Tokenization

**Example**:
```python
code = "print('Hello World')"

# Different tokenizers split differently
# Some preserve syntax better
# Important for code generation models
```

---

## 6. Common Patterns

### Pattern 1: Unknown Words

**Long/rare word**: "supercalifragilisticexpialidocious"

**How handled**:
```
BPE: Breaks into many subwords
WordPiece: Also breaks into subwords
Character-level: One token per character
```

**Result**: No [UNK] needed with subword tokenization!

---

### Pattern 2: Punctuation

**Text**: "Hello, how are you?"

**Tokenization**:
```
GPT-2: ["Hello", ",", "Ġhow", "Ġare", "Ġyou", "?"]
       (Ġ = space marker)

BERT: ["hello", ",", "how", "are", "you", "?"]
      (lowercase)
```

---

### Pattern 3: Whitespace

**Handling spaces**:
```
GPT-2: Encodes spaces as "Ġ" prefix
BERT: Implicit (WordPiece boundaries)
```

---

## 7. Code Implementation

### Basic Tokenization
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('gpt2')

# Tokenize
tokens = tokenizer.tokenize("Hello world")
# ['Hello', 'Ġworld']

# Encode (text → IDs)
ids = tokenizer.encode("Hello world")
# [15496, 995]

# Decode (IDs → text)
text = tokenizer.decode(ids)
# 'Hello world'
```

---

### With Special Tokens
```python
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Get special token info
print(tokenizer.cls_token)  # '[CLS]'
print(tokenizer.sep_token)  # '[SEP]'
print(tokenizer.pad_token)  # '[PAD]'

# Encode with special tokens
encoded = tokenizer.encode_plus(
    "Hello",
    add_special_tokens=True,
    padding='max_length',
    max_length=10,
    return_tensors='pt'
)
```

---

### Check Token Limits
```python
text = "Very long text..." * 1000
tokens = tokenizer.encode(text)

print(f"Token count: {len(tokens)}")
print(f"Exceeds limit: {len(tokens) > 1024}")

# Truncate if needed
if len(tokens) > 1024:
    tokens = tokens[:1024]
```

---

## 8. Key Takeaways

**Tokenization is crucial**:
```
✓ Converts text to numbers
✓ Enables model processing
✓ Affects model performance
✓ Different methods for different use cases
```

**Best Practices**:
```
1. Use pretrained tokenizer with pretrained model
2. Be aware of token limits
3. Handle special tokens correctly
4. Consider language-specific needs
5. Test with your data
```

**Common Issues**:
```
❌ Mixing tokenizers (BERT tokenizer + GPT model)
❌ Ignoring token limits
❌ Not handling special tokens
❌ Forgetting to decode properly
```

---

## 9. Terminology

| Term | Definition |
|------|------------|
| **Token** | Basic unit of text (word, subword, character) |
| **Vocabulary** | Set of all possible tokens |
| **BPE** | Byte-Pair Encoding algorithm |
| **WordPiece** | Google's tokenization algorithm |
| **Subword** | Part of a word (e.g., "play" in "playing") |
| **[CLS]** | Classification/start token (BERT) |
| **[SEP]** | Separator/end token (BERT) |
| **[PAD]** | Padding token (fill sequences) |
| **[UNK]** | Unknown token (rare words) |
| **Context Window** | Max tokens model can process |
| **Attention Mask** | Binary mask (1=real token, 0=padding) |

---

## 10. Comparison Table

| Feature | GPT-2 (BPE) | BERT (WordPiece) |
|---------|-------------|------------------|
| **Algorithm** | Byte-Pair Encoding | WordPiece |
| **Vocab Size** | ~50k | ~30k |
| **Special Tokens** | Minimal | [CLS], [SEP], [PAD] |
| **Case** | Case-sensitive | Optional |
| **Spaces** | Ġ prefix | Implicit |
| **Subword Marker** | None | ## prefix |
| **Context Limit** | 1024 tokens | 512 tokens |
| **Best For** | Generation | Understanding |

---

## 11. Next Steps (Day 4)

**Plan**:
```
1. Word Embeddings
   - How tokens become vectors
   - Embedding dimensions
   - Semantic meaning

2. Position Encodings
   - Why needed
   - Sinusoidal encoding
   - Learned vs fixed

3. Input Preparation
   - Complete pipeline
   - Batch processing
   - Attention masks
```

---

## Resources

**Documentation**:
- HuggingFace: https://huggingface.co/learn/llm-course/chapter2/4
- Transformers library: https://huggingface.co/docs/transformers

**Code**:
- Day_3_Tokenization.py (included)

---

## Experiment Results Summary

**Tested**:
- ✅ 5 different text types
- ✅ 2 tokenizer types (GPT-2, BERT)
- ✅ Special token handling
- ✅ Token limit scenarios

**Learned**:
- ✅ Tokenizers split text differently
- ✅ Subword tokenization handles unknown words
- ✅ Special tokens have specific purposes
- ✅ Context windows are limited
- ✅ Encoding is reversible

---

**Day 3 Complete!** ✅
