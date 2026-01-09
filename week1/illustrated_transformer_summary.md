# The Illustrated Transformer - Summary

**Source**: Jay Alammar - "The Illustrated Transformer"  
**Day 2 - AI Learning**

---

## 1. High-Level Architecture (高层架构)

### Black Box View
```
Input Sentence → [Transformer] → Output Translation
Example: "merci" → [Transformer] → "thanks"
```

### Two Main Components
```
┌─────────────────────────────────────┐
│         TRANSFORMER                 │
│                                     │
│  ┌──────────┐      ┌──────────┐   │
│  │ ENCODER  │  →   │ DECODER  │   │
│  │ (Stack)  │      │ (Stack)  │   │
│  └──────────┘      └──────────┘   │
└─────────────────────────────────────┘
```

### Stack Structure
- **6 Encoders** stacked (可以调整数量)
- **6 Decoders** stacked (same number)
- All encoders identical structure (but different weights)
- All decoders identical structure (but different weights)

---

## 2. Encoder Structure (编码器结构)

Each encoder has **2 sub-layers**:

```
Input
  ↓
┌─────────────────────┐
│ Self-Attention      │ ← Layer 1
└─────────────────────┘
  ↓
┌─────────────────────┐
│ Feed-Forward        │ ← Layer 2
│ Neural Network      │
└─────────────────────┘
  ↓
Output to next encoder
```

### Key Points:
- **Self-Attention**: Helps encoder look at other words while encoding current word
- **Feed-Forward**: Same network applied to each position independently
- **Parallel Processing**: Each word flows through its own path

---

## 3. Decoder Structure (解码器结构)

Each decoder has **3 sub-layers**:

```
Target Input
  ↓
┌─────────────────────────┐
│ Masked Self-Attention   │ ← Layer 1 (防止看到future)
└─────────────────────────┘
  ↓
┌─────────────────────────┐
│ Encoder-Decoder         │ ← Layer 2 (cross-attention)
│ Attention               │
└─────────────────────────┘
  ↓
┌─────────────────────────┐
│ Feed-Forward Network    │ ← Layer 3
└─────────────────────────┘
  ↓
Output
```

### Key Difference:
- **Masked Self-Attention**: Only attends to earlier positions (not future)
- **Cross-Attention**: Queries from decoder, Keys/Values from encoder

---

## 4. Input Processing (输入处理)

### Step 1: Word Embeddings (词嵌入)
```
Word → Embedding Vector (512 dimensions)

Example:
"Thinking" → [0.12, 0.45, -0.23, ..., 0.89]  (512 numbers)
"Machines" → [0.34, -0.12, 0.67, ..., 0.12]  (512 numbers)
```

### Step 2: Positional Encoding (位置编码)
```
Embedding + Positional Encoding = Final Input

Why? Attention has no sense of position!
"dog bites man" vs "man bites dog" need different meanings
```

**Positional Encoding Formula**:
```python
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

where:
- pos: position in sequence (0,1,2,...)
- i: dimension index
- d_model: 512
```

---

## 5. Self-Attention in Detail (自注意力详解)

### The Problem Self-Attention Solves

```
Sentence: "The animal didn't cross the street because it was too tired"

Question: What does "it" refer to?
- The street? or
- The animal? ✓

Self-attention helps "it" look at "animal" for context
```

### How It Works - 6 Steps

**Step 1: Create Q, K, V vectors** (创建查询、键、值向量)

```python
# For each word embedding (512-dim):
Q = Embedding × W_Q  # Query (64-dim)
K = Embedding × W_K  # Key (64-dim)
V = Embedding × W_V  # Value (64-dim)

# W_Q, W_K, W_V are learned weight matrices
```

**Step 2: Calculate Scores** (计算分数)
```python
# For word "Thinking", calculate against all words:
score_1 = Q_thinking · K_thinking  # Dot product
score_2 = Q_thinking · K_machines
...
```

**Step 3: Scale** (缩放)
```python
scaled_scores = scores / √(d_k)
             = scores / √64
             = scores / 8
```

**Step 4: Softmax** (归一化)
```python
attention_weights = softmax(scaled_scores)
# Converts to probabilities (sum = 1)

Example:
[0.5, 0.3, 0.15, 0.05] ← How much to focus on each word
```

**Step 5: Multiply by Values** (乘以值向量)
```python
weighted_values = attention_weights × V
```

**Step 6: Sum** (求和)
```python
output = sum(weighted_values)
```

### Matrix Form (矩阵形式)

**All 6 steps in one formula**:
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**Dimensions**:
```
X: [seq_len, 512]     Input embeddings
Q: [seq_len, 64]      Queries
K: [seq_len, 64]      Keys  
V: [seq_len, 64]      Values
Output: [seq_len, 64]
```

---

## 6. Multi-Head Attention (多头注意力)

### Why Multiple Heads?

**Problem**: One attention head = one type of relationship

**Solution**: 8 parallel attention heads

```
Head 1 → Focuses on: Subject-verb relations
Head 2 → Focuses on: Adjective-noun relations
Head 3 → Focuses on: Long-range dependencies
...
Head 8 → Focuses on: Different aspect
```

### Example:
```
Sentence: "The animal didn't cross the street because it was too tired"

Head 1: "it" attends to "animal" (0.9)
Head 2: "it" attends to "tired" (0.7)
→ Different heads capture different relationships!
```

### Implementation

```python
# 8 separate attention calculations in parallel
for i in range(8):
    Q_i = X @ W_Q_i  # Each head has own weights
    K_i = X @ W_K_i
    V_i = X @ W_V_i
    
    head_i = Attention(Q_i, K_i, V_i)  # [seq_len, 64]

# Concatenate all heads
multi_head = Concat(head_1, ..., head_8)  # [seq_len, 512]

# Final projection
output = multi_head @ W_O  # [seq_len, 512]
```

**Dimensions**:
- Each head: 64 dimensions
- 8 heads: 8 × 64 = 512 total
- Back to original embedding size!

---

## 7. Residual Connections (残差连接)

### Structure
```
Input
  ↓
  ├─→ Self-Attention ─→ +  ← Add input to output
  ↓                     ↓
  └─────────────────────┘
  ↓
Layer Normalization
  ↓
  ├─→ Feed-Forward ─→ +
  ↓                  ↓
  └──────────────────┘
  ↓
Layer Normalization
```

### Formula
```python
# Around self-attention
output = LayerNorm(x + SelfAttention(x))

# Around feed-forward
output = LayerNorm(x + FeedForward(x))
```

### Purpose
- Helps gradient flow (梯度流动)
- Enables deeper networks
- Stabilizes training

---

## 8. Feed-Forward Network (前馈网络)

### Structure
```python
FFN(x) = max(0, x·W1 + b1)·W2 + b2

# Two linear layers with ReLU in between
```

### Dimensions
```
Input:  [seq_len, 512]
Hidden: [seq_len, 2048]  ← Expand 4x
Output: [seq_len, 512]   ← Compress back
```

### Key Point
- Applied to **each position independently**
- Same network for all positions
- No interaction between positions (unlike attention)

---

## 9. Decoder Specific Features

### Masked Self-Attention (掩码自注意力)

**Purpose**: Prevent looking at future tokens

```python
# Causal mask
mask = [
    [1, 0, 0, 0],  # Position 0 sees only position 0
    [1, 1, 0, 0],  # Position 1 sees 0,1
    [1, 1, 1, 0],  # Position 2 sees 0,1,2
    [1, 1, 1, 1]   # Position 3 sees all
]

# Apply mask before softmax
scores = scores + (1 - mask) * (-1e9)  # Mask = -inf
```

### Encoder-Decoder Attention (交叉注意力)

```python
# Different from self-attention!
Q = from decoder layer below
K = from encoder output  ← Different source!
V = from encoder output  ← Different source!

# Decoder queries what it needs from encoder
```

---

## 10. Output Generation (输出生成)

### Final Layers
```
Decoder Output
  ↓
┌─────────────────┐
│ Linear Layer    │ ← Projects to vocabulary size
└─────────────────┘
  ↓
┌─────────────────┐
│ Softmax         │ ← Converts to probabilities
└─────────────────┘
  ↓
Word Probabilities
```

### Example
```python
# Decoder outputs: [seq_len, 512]
# Linear projects: [seq_len, 10000]  ← Vocab size
# Softmax: [seq_len, 10000]  ← Probabilities

# Pick highest probability word
output_word = argmax(probabilities)
```

---

## 11. Training Process (训练过程)

### Loss Function

**Cross-Entropy Loss** between:
- **Predicted distribution**: Model output
- **Target distribution**: Actual correct translation

```python
# Example: Translating "merci" → "thanks"

Predicted: [0.1, 0.05, 0.3, 0.5, 0.03, 0.02]  ← Model output
Target:    [0,   0,    0,   1,   0,    0   ]  ← "thanks" = position 3

Loss = CrossEntropy(Predicted, Target)
```

### Training Step
```
1. Feed input sentence through encoder
2. Feed target sentence through decoder
3. Compare output with expected output
4. Calculate loss
5. Backpropagate
6. Update weights
7. Repeat
```

### Decoding Methods

**Greedy Decoding**:
```python
# Always pick highest probability
word = argmax(probabilities)
```

**Beam Search**:
```python
# Keep top K candidates at each step
# Explore multiple paths
# Pick best complete sequence

beam_size = 2
# Keep 2 best partial translations at each step
```

---

## 12. Key Dimensions Summary

### Standard Transformer (Original Paper)

```
d_model = 512        # Embedding dimension
d_k = d_v = 64       # Per-head dimension  
num_heads = 8        # Attention heads
d_ff = 2048          # Feed-forward hidden
num_layers = 6       # Encoder/decoder layers
vocab_size = ~30000  # Vocabulary size
```

### Calculation
```
d_k × num_heads = 64 × 8 = 512 = d_model ✓
```

---

## 13. Information Flow (信息流动)

### Encoder
```
Input Text: "The cat sat"
  ↓
Embeddings: [3, 512]
  ↓
+ Positional Encoding
  ↓
Encoder Layer 1:
  Self-Attention → [3, 512]
  Feed-Forward → [3, 512]
  ↓
Encoder Layer 2-6:
  (same structure)
  ↓
Final Encoding: [3, 512]
```

### Decoder
```
Target Text: "Le chat"
  ↓
Embeddings: [2, 512]
  ↓
+ Positional Encoding
  ↓
Decoder Layer 1:
  Masked Self-Attention → [2, 512]
  Cross-Attention (to encoder) → [2, 512]
  Feed-Forward → [2, 512]
  ↓
Decoder Layer 2-6:
  (same structure)
  ↓
Linear + Softmax
  ↓
Output Probabilities: [2, vocab_size]
```

---

## 14. Key Terminology (关键术语)

| English | 中文 | Meaning |
|---------|------|---------|
| **Encoder** | 编码器 | Processes input sequence |
| **Decoder** | 解码器 | Generates output sequence |
| **Self-Attention** | 自注意力 | Attend to same sequence |
| **Cross-Attention** | 交叉注意力 | Decoder attends to encoder |
| **Multi-Head** | 多头 | Multiple parallel attentions |
| **Query (Q)** | 查询 | What we're looking for |
| **Key (K)** | 键 | What's available |
| **Value (V)** | 值 | Actual content |
| **Positional Encoding** | 位置编码 | Add position information |
| **Residual Connection** | 残差连接 | Skip connection (x + F(x)) |
| **Layer Normalization** | 层归一化 | Normalize across features |
| **Feed-Forward** | 前馈网络 | Position-wise MLP |
| **Masking** | 掩码 | Block future positions |
| **Embedding** | 嵌入 | Word to vector |

---

## 15. Why Transformer Works Better

### vs RNN/LSTM

| Feature | RNN/LSTM | Transformer |
|---------|----------|-------------|
| **Processing** | Sequential ❌ | Parallel ✓ |
| **Long-range** | Gradient vanishing ❌ | Direct connections ✓ |
| **Training Speed** | Slow ❌ | Fast ✓ |
| **Max Distance** | Limited | Any distance ✓ |

### Key Advantages

1. **Parallelization** (并行化)
   - All positions processed simultaneously
   - Much faster on GPUs

2. **Long-range Dependencies** (长距离依赖)
   - Direct attention between any two positions
   - No gradient vanishing

3. **Interpretability** (可解释性)
   - Can visualize attention weights
   - See what model focuses on

---

## 16. Visual Summary

### Complete Architecture
```
INPUT: "The cat sat"

┌─────────────────────────────────────┐
│ ENCODER STACK                       │
├─────────────────────────────────────┤
│ Input Embedding + Positional        │
│         ↓                            │
│ ┌──────────────────────┐            │
│ │ Encoder Layer 1-6    │            │
│ │  - Self-Attention    │            │
│ │  - Feed-Forward      │            │
│ │  (with residuals)    │            │
│ └──────────────────────┘            │
│         ↓                            │
│ Encoder Output [3, 512]             │
└─────────────────────────────────────┘
              ↓ K, V
┌─────────────────────────────────────┐
│ DECODER STACK                       │
├─────────────────────────────────────┤
│ Target Embedding + Positional       │
│         ↓                            │
│ ┌──────────────────────┐            │
│ │ Decoder Layer 1-6    │            │
│ │  - Masked Self-Attn  │            │
│ │  - Cross-Attention   │← K,V       │
│ │  - Feed-Forward      │            │
│ │  (with residuals)    │            │
│ └──────────────────────┘            │
│         ↓                            │
│ Linear + Softmax                    │
└─────────────────────────────────────┘
              ↓
OUTPUT: "Le chat était"
```

---

## 17. Core Concepts to Remember

### 1. **Attention is Key**
```
Allows model to focus on relevant parts
Q, K, V mechanism enables this
```

### 2. **Multi-Head = Multiple Perspectives**
```
Different heads learn different relationships
8 heads in parallel
```

### 3. **Position Matters**
```
Positional encoding essential
Without it: "dog bites man" = "man bites dog"
```

### 4. **Residuals Enable Depth**
```
x + F(x) allows gradient flow
Can stack 6+ layers
```

### 5. **Encoder-Decoder Architecture**
```
Encoder: Understand input
Decoder: Generate output
Cross-attention: Connect them
```

---

## Self-Check Questions

1. What are the two main components of Transformer?
2. How many sub-layers in an encoder? In a decoder?
3. What do Q, K, V represent?
4. Why scale by √d_k in attention?
5. Why do we need multi-head attention?
6. What's the purpose of positional encoding?
7. What's the difference between self-attention and cross-attention?
8. Why does decoder use masking?
9. What are residual connections for?
10. How does output generation work?

---

## Quick Reference

**Standard Dimensions**:
- Embedding: 512
- Heads: 8
- Per-head: 64
- Feed-forward: 2048
- Layers: 6

**Key Formula**:
```
Attention(Q,K,V) = softmax(QK^T/√d_k)V
```

**Architecture**:
- Encoder: Self-Attention + FFN
- Decoder: Masked Self-Attention + Cross-Attention + FFN
- Both: Residuals + LayerNorm

---

**Day 2 Complete!** ✅

This visual walkthrough makes Transformer architecture much clearer than reading the paper alone!
