# Transformer & Attention - Concise Knowledge Summary

**Source**: 3Blue1Brown - Attention in Transformers
**Day 1 Notes**

---

## 1. Core Problem & Solution

**Problem**: How can each word understand its context?

**RNN Limitation**:
- Sequential processing (顺序处理)
- Gradient vanishing in long sequences (梯度消失)

**Transformer Solution**:
- Parallel processing (并行处理)
- Direct attention between any two positions
- No gradient vanishing

---

## 2. Attention Mechanism

**Definition**: Allows each position to focus on relevant parts of input

**Formula**:
```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
```

**Process**:
1. Compute similarity scores: Q·K^T
2. Scale: divide by √d_k
3. Normalize: softmax
4. Weight values: multiply by V

---

## 3. Query, Key, Value (QKV)

**Analogy**: Library search system

| Component | Meaning | Purpose |
|-----------|---------|---------|
| **Query (Q)** (查询) | "What I'm looking for" | What information this word wants |
| **Key (K)** (键) | "What I can provide" | What information each word offers |
| **Value (V)** (值) | "My actual content" | The actual representation to pass |

**Creation**:
```python
Q = Input @ W_Q  # W_Q, W_K, W_V are learnable
K = Input @ W_K
V = Input @ W_V
```

**Dimensions**:
- d_model: 512 (embedding dimension)
- d_k = d_v: 64 (per-head dimension)

---

## 4. Mathematical Details

**Step 1: Dot Product (点积)**
- Measures similarity between vectors
- Q·K^T produces [seq_len × seq_len] matrix

**Step 2: Scaling (缩放)**
- Divide by √d_k
- Prevents large values that cause gradient issues
- Stabilizes softmax

**Step 3: Softmax (归一化)**
- Converts to probability distribution
- All weights sum to 1
- Creates attention weights

**Step 4: Apply to Values (加权求和)**
- Weighted sum: attention_weights @ V
- Produces context-enriched representation

---

## 5. Multi-Head Attention

**Why**: Single head learns only ONE type of relationship

**Solution**: 8 parallel heads learning different aspects

**What different heads learn**:
- Head 1: Syntactic relations (句法关系)
- Head 2: Semantic relations (语义关系)
- Head 3: Long-range dependencies (长距离依赖)
- Head 4: Local context (局部上下文)

**Implementation**:
```python
for i in range(num_heads):
    Q_i = X @ W_Q_i
    K_i = X @ W_K_i
    V_i = X @ W_V_i
    head_i = Attention(Q_i, K_i, V_i)

output = Concat(heads) @ W_O
```

**Dimensions**:
- Total: d_model = 512
- Per head: d_k = 512/8 = 64

---

## 6. Positional Encoding

**Problem**: Attention is permutation invariant (位置无关)
- "dog bites man" = "man bites dog" without position info

**Solution**: Add unique encoding to each position

**Formula**:
```python
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Why sinusoidal**:
- Unique for each position
- Generalizes to unseen lengths
- Enables relative position learning
- Values bounded in [-1, 1]

---

## 7. Transformer Architecture

**Encoder Block**:
```
Input + Positional Encoding
    ↓
Multi-Head Self-Attention
    ↓
Add & Normalize (残差 + 层归一化)
    ↓
Feed-Forward Network
    ↓
Add & Normalize
    ↓
Output
```

**Decoder Block**:
```
Target + Positional Encoding
    ↓
Masked Self-Attention (masked 自注意力)
    ↓
Add & Normalize
    ↓
Cross-Attention (交叉注意力, attends to encoder)
    ↓
Add & Normalize
    ↓
Feed-Forward Network
    ↓
Add & Normalize
    ↓
Output
```

**Key Components**:
- **Residual Connection** (残差连接): x + SubLayer(x)
- **Layer Norm** (层归一化): Normalizes across features
- **Feed-Forward**: 2-layer MLP, expands then compresses
  - [d_model=512] → [d_ff=2048] → [d_model=512]

---

## 8. Masking

**Causal Mask** (因果掩码) in decoder:
```python
# Prevents attending to future tokens
mask = [
    [1, 0, 0, 0],  # Position 0 sees only position 0
    [1, 1, 0, 0],  # Position 1 sees 0,1
    [1, 1, 1, 0],  # Position 2 sees 0,1,2
    [1, 1, 1, 1]   # Position 3 sees all
]
```

**Purpose**: Autoregressive generation (自回归生成)

---

## 9. Key Properties

**Attention is**:
- **Dynamic** (动态): Changes based on context
- **Content-based** (基于内容): Depends on input, not position
- **Parallel** (并行): All positions computed simultaneously
- **Interpretable** (可解释): Can visualize attention weights

**Complexity**:
- Time: O(n² × d)
- Space: O(n²)
- Problem for very long sequences (n > 10,000)

---

## 10. Transformer vs RNN/LSTM

| Feature | RNN/LSTM | Transformer |
|---------|----------|-------------|
| Processing | Sequential | Parallel ✓ |
| Long-range | Gradient vanishing ✗ | Direct connections ✓ |
| Speed | Slow | Fast ✓ |
| Memory | O(n) ✓ | O(n²) |
| Position | Implicit ✓ | Needs encoding |

---

## 11. Critical Insights

1. **Self-Attention = Dynamic Contextualization**
   - Same word gets different representations in different contexts

2. **Scaling (√d_k) is Essential**
   - Without it: gradient instability
   - Normalizes dot product variance

3. **Multi-Head = Multiple Perspectives**
   - Not redundancy, learns different relationship types

4. **Positional Encoding Required**
   - Pure attention has no position awareness
   - Sinusoidal allows any sequence length

5. **Residuals Enable Deep Networks**
   - Gradient can flow through skip connections
   - Allows training 100+ layers

6. **Layer Depth = Abstraction Hierarchy**
   - Early layers: local syntax
   - Deep layers: high-level semantics

---

## 12. Key Terminology

**Core Concepts**:
- Self-Attention (自注意力): Attention within same sequence
- Cross-Attention (交叉注意力): Between encoder-decoder
- Multi-Head (多头注意力): Parallel attention computations

**Mathematics**:
- Dot Product (点积): Q·K similarity measure
- Softmax (归一化): Convert to probabilities
- Scaling Factor (缩放因子): √d_k

**Architecture**:
- Encoder (编码器): Processes input
- Decoder (解码器): Generates output
- Residual (残差): Skip connection
- Layer Norm (层归一化): Normalization technique

**Training**:
- Masking (掩码): Block certain positions
- Pre-training (预训练): Learn on large data
- Fine-tuning (微调): Adapt to task

---

## 13. Common Applications

- **Translation** (翻译): English → French
- **Question Answering** (问答): Extract answers from context
- **Summarization** (摘要): Long text → short summary
- **Conversational AI** (对话AI): ChatGPT
- **Code Generation** (代码生成): GitHub Copilot
- **Image Understanding** (图像理解): Vision Transformers

---

## 14. Important Equations

**Attention**:
```
Attention(Q,K,V) = softmax(QK^T/√d_k)V
```

**Multi-Head**:
```
MultiHead(Q,K,V) = Concat(head_1,...,head_h)W_O
where head_i = Attention(QW_Q^i, KW_K^i, VW_V^i)
```

**Positional Encoding**:
```
PE(pos,2i) = sin(pos/10000^(2i/d_model))
PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
```

**Layer**:
```
Output = LayerNorm(x + Sublayer(x))
```

---

## Quick Reference

**Standard Hyperparameters** (GPT/BERT-style):
- d_model: 512 or 768
- num_heads: 8 or 12
- d_ff: 2048 or 3072
- num_layers: 6 or 12
- d_k = d_v: d_model / num_heads

**Typical Architecture**:
- 6 encoder layers
- 6 decoder layers (if encoder-decoder)
- 8 attention heads per layer
- 2048 feed-forward dimension

---

## Self-Check Questions

1. What does Q·K^T compute?
2. Why scale by √d_k?
3. What does softmax do to attention scores?
4. Why multiple attention heads?
5. Why is positional encoding needed?
6. Difference between self-attention and cross-attention?
7. Purpose of residual connections?
8. What is causal masking for?

---

**Next**: Day 2 - Tokenization & Embeddings
