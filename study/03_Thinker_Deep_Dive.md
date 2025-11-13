# Thinker: The Core Language Model

## 🎯 Key Takeaways (TL;DR)

- **What**: Thinker is a decoder-only transformer (like GPT)
- **Why**: Enables autoregressive text generation and multimodal understanding
- **How**: 4 transformer blocks with RoPE, SwiGLU, RMSNorm, and pre-norm architecture
- **Key Insight**: Pre-norm architecture enables stable deep training (100+ layers possible)
- **Common Mistake**: Forgetting causal mask in attention (allows seeing future tokens)
- **Shape Flow**: `[B, T] → [B, T, 256] → ... → [B, T, 5000]` (batch, tokens, vocab)

**📖 Reading Guide**:
- **Quick Read**: 5 minutes (overview only)
- **Standard Read**: 20 minutes (full document)
- **Deep Dive**: 60 minutes (read + code + exercises)

## What is Thinker?

**Thinker** is μOmni's core language model - a decoder-only transformer that processes unified multimodal embeddings.

Think of it as the "brain" that:
- Understands text, images, and audio (after encoding)
- Generates text responses
- Maintains context across the conversation

## Architecture

```
Input (Text/Image/Audio embeddings)
    ↓
Token/Embedding Layer
    ↓
┌─────────────────────┐
│  Transformer Block  │
│  ┌───────────────┐  │
│  │ RMSNorm       │  │
│  └───────┬───────┘  │
│          │          │
│  ┌───────▼───────┐  │
│  │ Self-Attention│  │
│  └───────┬───────┘  │
│          │          │
│  ┌───────▼───────┐  │
│  │ RMSNorm       │  │
│  └───────┬───────┘  │
│          │          │
│  ┌───────▼───────┐  │
│  │    MLP        │  │
│  │  (SwiGLU)     │  │
│  └───────────────┘  │
└─────────────────────┘
    ↓ (repeat N times)
Output Head
    ↓
Token Predictions
```

### Diagram 1: Thinker Complete Architecture

```mermaid
graph TD
    Input[Input Embeddings<br/>Multimodal] --> Embed[Token Embedding<br/>vocab_size → d_model]
    Embed --> Block1[Transformer Block 1]
    Block1 --> Block2[Transformer Block 2]
    Block2 --> Dots[...]
    Dots --> BlockN[Transformer Block N]
    BlockN --> Norm[Final RMSNorm]
    Norm --> Head[Output Head<br/>d_model → vocab_size]
    Head --> Logits[Token Logits]
    
    Block1 -.->|RMSNorm + Attention + MLP| Block1
    Block2 -.->|RMSNorm + Attention + MLP| Block2
    BlockN -.->|RMSNorm + Attention + MLP| BlockN
    
    style Embed fill:#3498db
    style Block1 fill:#9b59b6
    style Block2 fill:#9b59b6
    style BlockN fill:#9b59b6
    style Head fill:#e74c3c
    style Logits fill:#27ae60
```

**Explanation**: Thinker processes unified embeddings through N transformer blocks, each containing normalization, attention, and feedforward layers, producing token predictions via the output head.

## Shape Flow Through Thinker

Understanding tensor shapes is crucial for debugging. Here's how data flows:

```
Input:  [1, 10]           # (batch=1, tokens=10)
  ↓
Embed:  [1, 10, 256]      # (batch, tokens, d_model=256)
  ↓
Block1: [1, 10, 256]      # Same shape (residual connections)
  ↓
Block2: [1, 10, 256]
  ↓
Block3: [1, 10, 256]
  ↓
Block4: [1, 10, 256]
  ↓
Final Norm: [1, 10, 256]
  ↓
Head:   [1, 10, 5000]     # (batch, tokens, vocab_size=5000)
```

**Key Points**:
- Batch dimension stays constant: `1`
- Sequence length stays constant: `10`
- Hidden dimension stays constant: `256` (until output head)
- Only output head expands to vocab size: `5000`

## Key Components

### 1. Embedding Layer

Converts token IDs to dense vectors:

```python
# From omni/thinker.py
self.tok_emb = nn.Embedding(vocab_size, d_model)
# vocab_size = 5000 (number of unique tokens)
# d_model = 256 (embedding dimension)
```

**Example**:
```python
token_id = 1234
embedding = tok_emb(token_id)  # Shape: (256,)
```

### Try It Yourself

```python
# Run this to see embeddings in action
import torch
import torch.nn as nn

# Create embedding layer (like Thinker)
vocab_size = 5000
d_model = 256
tok_emb = nn.Embedding(vocab_size, d_model)

# Test with token IDs
token_ids = torch.tensor([[1, 1234, 5678]])  # (batch=1, seq_len=3)
embeddings = tok_emb(token_ids)

print(f"Input shape: {token_ids.shape}")
print(f"Output shape: {embeddings.shape}")
# Expected: Input shape: torch.Size([1, 3])
# Expected: Output shape: torch.Size([1, 3, 256])
```

### 2. Transformer Blocks

Each block contains:
- **RMSNorm**: Normalization layer
- **Self-Attention**: Looks at all positions
- **MLP**: Feedforward network

### Diagram 2: Transformer Block Structure

```mermaid
graph TD
    Input[Input x] --> Norm1[RMSNorm]
    Norm1 --> Attn[Self-Attention<br/>Multi-Head]
    Attn --> Add1[Add: x + attn_out]
    Input --> Add1
    Add1 --> Norm2[RMSNorm]
    Norm2 --> MLP[MLP<br/>SwiGLU]
    MLP --> Add2[Add: x + mlp_out]
    Add1 --> Add2
    Add2 --> Output[Output]
    
    Attn -.->|Q, K, V| QKV[QKV Projection]
    QKV -.->|RoPE| Rope[Rotary Position<br/>Encoding]
    Rope -.->|Attention| Attn
    
    style Norm1 fill:#3498db
    style Attn fill:#e74c3c
    style Norm2 fill:#3498db
    style MLP fill:#9b59b6
```

**Explanation**: Each transformer block uses pre-norm architecture with residual connections. Self-attention processes all positions, MLP processes each position independently.

#### RMSNorm (Root Mean Square Normalization)

Simpler than LayerNorm, normalizes by RMS:

```python
def rms_norm(x, eps=1e-6):
    rms = torch.sqrt(torch.mean(x**2) + eps)
    return x / rms
```

#### Self-Attention

The heart of transformers - allows model to focus on relevant parts:

```python
# Simplified attention
def attention(query, key, value):
    # Compute similarity scores
    scores = query @ key.T / sqrt(d_k)
    # Apply softmax (probabilities)
    weights = softmax(scores)
    # Weighted sum
    output = weights @ value
    return output
```

### Progressive Example: Understanding Attention

#### Level 1: Single Token (Simplest)
```python
# Just one token - no attention needed
q = torch.randn(64)  # Single query vector
k = torch.randn(64)  # Single key vector
score = q @ k  # Scalar similarity
# Output: Single number
```

#### Level 2: Sequence (Basic Attention)
```python
# Multiple tokens - attention between them
seq_len = 10
q = torch.randn(seq_len, 64)  # 10 queries
k = torch.randn(seq_len, 64)  # 10 keys
v = torch.randn(seq_len, 64)  # 10 values

scores = q @ k.T / (64 ** 0.5)  # (10, 10) similarity matrix
weights = torch.softmax(scores, dim=-1)  # (10, 10) attention weights
output = weights @ v  # (10, 64) weighted values
```

#### Level 3: Batched (Real Usage)
```python
# With batch dimension (like in Thinker)
batch_size = 8
seq_len = 10
d_model = 256
n_heads = 4
head_dim = 64

q = torch.randn(batch_size, n_heads, seq_len, head_dim)  # (8, 4, 10, 64)
k = torch.randn(batch_size, n_heads, seq_len, head_dim)  # (8, 4, 10, 64)
v = torch.randn(batch_size, n_heads, seq_len, head_dim)  # (8, 4, 10, 64)

scores = q @ k.transpose(-2, -1) / (head_dim ** 0.5)  # (8, 4, 10, 10)
weights = torch.softmax(scores, dim=-1)  # (8, 4, 10, 10)
output = weights @ v  # (8, 4, 10, 64)
```

### Diagram 3: Self-Attention Mechanism

```mermaid
graph LR
    Input[Input Tokens] --> Q[Query Q]
    Input --> K[Key K]
    Input --> V[Value V]
    
    Q --> Scores[Compute Scores<br/>Q @ K^T / √d_k]
    K --> Scores
    Scores --> Mask[Causal Mask<br/>Hide Future]
    Mask --> Softmax[Softmax<br/>Probabilities]
    Softmax --> Weights[Attention Weights]
    Weights --> Output[Weighted Sum<br/>Weights @ V]
    V --> Output
    
    style Q fill:#3498db
    style K fill:#3498db
    style V fill:#3498db
    style Weights fill:#e74c3c
    style Output fill:#27ae60
```

**Explanation**: Self-attention computes similarity scores between queries and keys, applies causal masking (for decoder), converts to probabilities, then uses these weights to combine value vectors.

**Multi-Head Attention**: Run attention multiple times in parallel:

```python
# 4 heads, each with 64 dimensions (256 total)
head1 = attention(q1, k1, v1)  # 64 dims
head2 = attention(q2, k2, v2)  # 64 dims
head3 = attention(q3, k3, v3)  # 64 dims
head4 = attention(q4, k4, v4)  # 64 dims
# Concatenate: 256 dims
```

### Diagram 4: Multi-Head Attention

```mermaid
graph TD
    Input[Input x] --> QProj[Q Projection]
    Input --> KProj[K Projection]
    Input --> VProj[V Projection]
    
    QProj --> SplitQ[Split into 4 heads]
    KProj --> SplitK[Split into 4 heads]
    VProj --> SplitV[Split into 4 heads]
    
    SplitQ --> Head1[Head 1<br/>64 dims]
    SplitK --> Head1
    SplitV --> Head1
    
    SplitQ --> Head2[Head 2<br/>64 dims]
    SplitK --> Head2
    SplitV --> Head2
    
    SplitQ --> Head3[Head 3<br/>64 dims]
    SplitK --> Head3
    SplitV --> Head3
    
    SplitQ --> Head4[Head 4<br/>64 dims]
    SplitK --> Head4
    SplitV --> Head4
    
    Head1 --> Concat[Concatenate]
    Head2 --> Concat
    Head3 --> Concat
    Head4 --> Concat
    Concat --> OutProj[Output Projection]
    OutProj --> Output[Output 256 dims]
    
    style Head1 fill:#3498db
    style Head2 fill:#3498db
    style Head3 fill:#3498db
    style Head4 fill:#3498db
    style Concat fill:#9b59b6
```

**Explanation**: Multi-head attention splits Q, K, V into multiple heads, processes each independently, then concatenates and projects back to the original dimension. Each head learns different attention patterns.

#### MLP (Multi-Layer Perceptron)

Feedforward network with SwiGLU activation:

```python
# SwiGLU: Swish-gated linear unit
def swiglu(x):
    gate = sigmoid(x)  # Gate
    return x * gate    # Gated output

# MLP structure
x → Linear → SwiGLU → Linear → output
```

### Diagram 6: MLP with SwiGLU

```mermaid
graph LR
    Input[Input x] --> Linear1[Linear 1<br/>d_model → d_ff]
    Linear1 --> Split[Split into 2 parts]
    Split --> Gate[Gate Part<br/>Sigmoid]
    Split --> Value[Value Part<br/>Linear]
    Gate --> Multiply[Element-wise<br/>Multiply]
    Value --> Multiply
    Multiply --> Linear2[Linear 2<br/>d_ff → d_model]
    Linear2 --> Output[Output]
    
    style Linear1 fill:#3498db
    style Gate fill:#9b59b6
    style Value fill:#3498db
    style Multiply fill:#e74c3c
    style Linear2 fill:#3498db
```

**Explanation**: SwiGLU splits the linear output into gate and value parts. The gate (sigmoid) controls how much of the value passes through, enabling more expressive transformations than standard ReLU.

### 3. RoPE (Rotary Position Embedding)

Handles sequence positions without adding separate position embeddings:

```python
# Rotates query/key vectors based on position
def apply_rope(x, position):
    # Rotate by angle based on position
    angle = position / (10000 ** (2 * i / d))
    rotated = rotate(x, angle)
    return rotated
```

### Diagram 5: RoPE Rotation

```mermaid
graph LR
    Q[Query Vector] --> RotQ[Rotate by θ_pos]
    K[Key Vector] --> RotK[Rotate by θ_pos]
    
    RotQ --> QRot[Rotated Q]
    RotK --> KRot[Rotated K]
    
    QRot --> Attn[Attention<br/>Q_rot @ K_rot^T]
    KRot --> Attn
    
    Pos[Position] --> Theta[Compute θ<br/>θ based on position]
    Theta --> RotQ
    Theta --> RotK
    
    style Q fill:#3498db
    style K fill:#3498db
    style QRot fill:#9b59b6
    style KRot fill:#9b59b6
    style Attn fill:#e74c3c
```

**Explanation**: RoPE rotates query and key vectors by an angle that depends on their position. This encodes relative positions in the attention scores, allowing the model to understand token order without explicit position embeddings.

**Why RoPE?**
- Better generalization to longer sequences
- Relative position encoding
- More efficient than absolute positions

### Concept Map: Attention Mechanism

```
Attention
├── Self-Attention (same sequence)
│   ├── Query (Q) - "What am I looking for?"
│   ├── Key (K) - "What do I have?"
│   └── Value (V) - "What information do I provide?"
├── Multi-Head (parallel attention)
│   └── 4 heads × 64 dims = 256 total
│   └── Each head learns different patterns
├── Causal Mask (decoder-only)
│   └── Hide future tokens (can't see what hasn't been generated)
└── RoPE (position encoding)
    └── Rotate Q/K by position angle
    └── Encodes relative positions in attention scores
```

### 4. Output Head

Converts hidden states to token predictions:

```python
self.head = nn.Linear(d_model, vocab_size)
# (256) → (5000) logits
```

## Advanced Features

### GQA (Grouped Query Attention)

Optional feature that shares key/value heads:

```
Standard: 4 query heads, 4 key heads, 4 value heads
GQA:      4 query heads, 2 key heads, 2 value heads (shared)
```

### Diagram 7: GQA vs Standard Attention

```mermaid
graph TD
    subgraph Standard["Standard Attention"]
        SQ[4 Query Heads] --> SA[4 Attention<br/>Computations]
        SK[4 Key Heads] --> SA
        SV[4 Value Heads] --> SA
    end
    
    subgraph GQA["Grouped Query Attention"]
        GQ[4 Query Heads] --> GA[4 Attention<br/>Computations]
        GK[2 Key Heads] --> GA
        GV[2 Value Heads] --> GA
        GK -.->|Shared| GK2[Same Keys<br/>for 2 Queries]
        GV -.->|Shared| GV2[Same Values<br/>for 2 Queries]
    end
    
    style SA fill:#3498db
    style GA fill:#27ae60
    style GK fill:#9b59b6
    style GV fill:#9b59b6
```

**Explanation**: GQA reduces KV cache size by sharing key/value heads across multiple query heads. This halves memory usage during inference while maintaining similar quality.

**Benefit**: Reduces memory usage while maintaining quality.

### MoE (Mixture of Experts)

Optional feature - replaces MLP with multiple "expert" networks:

```python
# Router selects top-2 experts per token
experts = [Expert1(), Expert2(), ..., Expert8()]
selected = router(x)  # Top 2 experts
output = selected[0](x) + selected[1](x)
```

### Diagram 8: Mixture of Experts

```mermaid
graph TD
    Input[Input x] --> Router[Router<br/>Select Top-2]
    Router --> E1[Expert 1]
    Router --> E2[Expert 2]
    Router --> E3[Expert 3]
    Router --> E4[Expert 4]
    Router --> E5[Expert 5]
    Router --> E6[Expert 6]
    Router --> E7[Expert 7]
    Router --> E8[Expert 8]
    
    E1 --> Weight1[Weight w1]
    E2 --> Weight2[Weight w2]
    E3 -.->|Not Selected| Skip
    E4 -.->|Not Selected| Skip
    E5 -.->|Not Selected| Skip
    E6 -.->|Not Selected| Skip
    E7 -.->|Not Selected| Skip
    E8 -.->|Not Selected| Skip
    
    Weight1 --> Sum[Weighted Sum]
    Weight2 --> Sum
    Sum --> Output[Output]
    
    style Router fill:#e74c3c
    style E1 fill:#3498db
    style E2 fill:#3498db
    style Sum fill:#9b59b6
```

**Explanation**: MoE uses a router to select the top-2 experts for each token. Only selected experts are activated, allowing larger model capacity (8 experts) while using compute for only 2 experts per token.

**Benefit**: Larger model capacity without proportional compute.

## Forward Pass

```python
def forward(self, x, embeddings=None):
    # Option 1: Token IDs
    if embeddings is None:
        x = self.tok_emb(x)  # (B, T, d_model)
    
    # Option 2: Raw embeddings (for multimodal)
    else:
        x = embeddings  # Already embeddings
    
    # Process through transformer blocks
    for block in self.blocks:
        x = block(x)  # (B, T, d_model)
    
    # Final normalization
    x = self.norm(x)
    
    # Predict next tokens
    logits = self.head(x)  # (B, T, vocab_size)
    return logits
```

## Training

### Objective: Next-Token Prediction

Given: `[BOS] "The cat sat"`
Predict: `"on"` (next token)

### Diagram 9: Training Flow

```mermaid
graph TD
    Text[Training Text] --> Tokenize[Tokenize]
    Tokenize --> Input["Input Tokens<br/>BOS, The, cat, sat"]
    Input --> Thinker[Thinker Forward]
    Thinker --> Logits["Logits<br/>vocab_size"]
    Logits --> Loss[Cross-Entropy Loss]
    Target["Target Tokens<br/>The, cat, sat, on"] --> Loss
    Loss --> Backward[Backward Pass]
    Backward --> Update[Update Weights]
    Update --> Next[Next Batch]
    Next --> Input
    
    style Thinker fill:#4a90e2
    style Loss fill:#e74c3c
    style Update fill:#27ae60
```

**Explanation**: Training uses next-token prediction - given a sequence, predict the next token. Cross-entropy loss measures prediction quality, and gradients update model weights.

```python
# Training example
input_ids = [1, 1234, 5678, 9012]  # [BOS, The, cat, sat]
target_ids = [1234, 5678, 9012, 3456]  # [The, cat, sat, on]

# Forward pass
logits = model(input_ids)  # (B, T, vocab_size)

# Calculate loss
loss = cross_entropy(logits, target_ids)
```

### Loss Function

Cross-entropy loss - measures how well predictions match targets:

```python
loss = -log(probability_of_correct_token)
```

## Inference (Generation)

### Autoregressive Generation

Generate one token at a time:

```python
def generate(prompt, max_length=64):
    tokens = tokenize(prompt)
    
    for _ in range(max_length):
        # Get predictions
        logits = model(tokens)
        # Get next token (greedy: highest probability)
        next_token = argmax(logits[:, -1, :])
```

### Diagram 10: Autoregressive Generation

```mermaid
graph LR
    Prompt[Prompt: 'Hello'] --> Step1[Step 1:<br/>Predict 'world']
    Step1 --> Tokens1[Tokens:<br/>Hello, world]
    Tokens1 --> Step2[Step 2:<br/>Predict '!']
    Step2 --> Tokens2[Tokens:<br/>Hello, world, !]
    Tokens2 --> Step3[Step 3:<br/>Predict EOS]
    Step3 --> Done[Generation<br/>Complete]
    
    Step1 -.->|KV Cache| Cache1[KV Cache]
    Step2 -.->|KV Cache| Cache2[KV Cache]
    Step3 -.->|KV Cache| Cache3[KV Cache]
    
    style Step1 fill:#3498db
    style Step2 fill:#3498db
    style Step3 fill:#3498db
    style Done fill:#27ae60
```

**Explanation**: Autoregressive generation produces one token at a time. Each step uses previously generated tokens (with KV caching for efficiency) to predict the next token until an end token is generated.
        # Append to sequence
        tokens.append(next_token)
        
        if next_token == EOS:
            break
    
    return detokenize(tokens)
```

### KV Caching

Speed up generation by caching previous computations:

```python
# First token: process full sequence
logits, kv_cache = model(prompt, use_cache=True)

# Subsequent tokens: only process new token
logits, kv_cache = model(new_token, kv_cache=kv_cache, use_cache=True)
```

**Benefit**: Much faster for long sequences!

## Configuration

From `configs/thinker_tiny.json`:

```json
{
  "vocab_size": 5000,      // Number of unique tokens
  "n_layers": 4,           // Number of transformer blocks
  "d_model": 256,          // Embedding dimension
  "n_heads": 4,            // Attention heads
  "d_ff": 1024,            // MLP hidden size
  "dropout": 0.1,          // Regularization
  "rope_theta": 10000,     // RoPE base frequency
  "ctx_len": 512,          // Maximum sequence length
  "use_swiglu": true,      // Use SwiGLU activation
  "use_gqa": false,        // Use grouped query attention
  "use_moe": false         // Use mixture of experts
}
```

## 🎬 Code Walkthrough: Attention Forward Pass

Let's trace through one attention computation step-by-step:

```python
# Step 1: Input
x = torch.randn(1, 10, 256)  # (batch=1, seq_len=10, dim=256)

# Step 2: Project to Q, K, V
qkv = self.qkv(x)  # (1, 10, 768) = 3 × 256
q, k, v = qkv.chunk(3, dim=-1)  # Each: (1, 10, 256)

# Step 3: Reshape for multi-head
q = q.view(1, 10, 4, 64)  # (batch, seq, heads, head_dim)
q = q.transpose(1, 2)     # (1, 4, 10, 64) - heads first
k = k.view(1, 10, 4, 64).transpose(1, 2)  # Same for K
v = v.view(1, 10, 4, 64).transpose(1, 2)  # Same for V

# Step 4: Apply RoPE (only to Q and K!)
q, k = self.rope(q, k, positions)  # Rotate by position

# Step 5: Compute attention scores
scores = q @ k.transpose(-2, -1) / math.sqrt(64)  # (1, 4, 10, 10)

# Step 6: Apply causal mask
mask = torch.triu(torch.ones(10, 10), diagonal=1).bool()
scores = scores.masked_fill(mask, float('-inf'))  # Hide future

# Step 7: Softmax to get attention weights
attn_weights = torch.softmax(scores, dim=-1)  # (1, 4, 10, 10)

# Step 8: Weighted sum of values
attn_output = attn_weights @ v  # (1, 4, 10, 64)

# Step 9: Concatenate heads and project
attn_output = attn_output.transpose(1, 2).contiguous()  # (1, 10, 4, 64)
attn_output = attn_output.view(1, 10, 256)  # (1, 10, 256)
output = self.out_proj(attn_output)  # (1, 10, 256)
```

**See actual code**:
- `omni/thinker.py:45-120` - Attention implementation
- `omni/utils.py:200-250` - RoPE implementation
- `omni/thinker.py:150-200` - Transformer block

## Code Walkthrough

Let's look at the actual implementation:

```python
# From omni/thinker.py

class ThinkerLM(nn.Module):
    def __init__(self, vocab_size, n_layers, d_model, ...):
        # Embedding layer
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, d_ff, ...)
            for _ in range(n_layers)
        ])
        
        # Output head
        self.head = nn.Linear(d_model, vocab_size)
    
    def forward(self, x, embeddings=None):
        # Handle both token IDs and embeddings
        if embeddings is None:
            x = self.tok_emb(x)
        else:
            x = embeddings
        
        # Process through blocks
        for block in self.blocks:
            x = block(x)
        
        # Predict tokens
        return self.head(x)
```

## Common Patterns

### 1. Pre-Normalization

Normalize BEFORE attention/MLP (not after):

```python
# Pre-norm (used in μOmni)
x = norm(x)
x = x + attention(x)

# Post-norm (older style)
x = x + attention(norm(x))
```

**Benefit**: More stable training.

### 2. Residual Connections

Add input to output (helps gradients flow):

```python
x = x + attention(x)  # Residual connection
```

### 3. Layer Scaling

Sometimes scale residual connections:

```python
x = x + alpha * attention(x)  # alpha < 1.0
```

## ⚡ Performance Tips

### Memory Optimization
- **Gradient Checkpointing**: `torch.utils.checkpoint` saves memory
- **Reduce Batch Size**: If OOM, reduce batch_size
- **Mixed Precision**: Use `torch.cuda.amp` for float16 training
- **Gradient Accumulation**: Simulate larger batches without memory cost

### Speed Optimization
- **KV Caching**: Enable during inference for 10x speedup
- **torch.compile()**: Use PyTorch 2.0+ compilation for 20-30% speedup
- **Batch Processing**: Process multiple sequences together
- **GPU Utilization**: Ensure GPU is fully utilized (check `nvidia-smi`)

### Example: Mixed Precision Training
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
for batch in dataloader:
    with autocast():
        logits = model(batch)
        loss = criterion(logits, targets)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## Performance Tips

1. **Batch Processing**: Process multiple sequences together
2. **KV Caching**: Cache attention states during generation
3. **Mixed Precision**: Use float16 for faster training
4. **Gradient Accumulation**: Simulate larger batches

## ⚠️ Common Pitfalls

Avoid these common mistakes:

1. **Forgetting Causal Mask**: Always mask future tokens in decoder
   ```python
   # WRONG: Can see future tokens
   scores = q @ k.T
   
   # CORRECT: Mask future tokens
   mask = torch.triu(torch.ones(T, T), diagonal=1)
   scores = scores.masked_fill(mask.bool(), float('-inf'))
   ```

2. **Wrong Shape Dimensions**: Check (B, T, D) format consistently
   ```python
   # Always verify shapes match
   assert x.shape == (batch_size, seq_len, d_model)
   ```

3. **RoPE on Wrong Tensors**: Only apply to Q and K, not V
   ```python
   # CORRECT: RoPE on Q and K only
   q, k = rope(q, k, positions)
   # WRONG: Don't apply to V
   # v = rope(v, positions)  # NO!
   ```

4. **Gradient Issues**: Use gradient clipping with RMSNorm
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

5. **Device Mismatch**: Ensure all tensors on same device
   ```python
   # Check device consistency
   assert x.device == model.device
   ```

## 🔍 Debugging Checklist: Training Not Working

When training fails, check these in order:

- [ ] **Data Loading**: Are batches correct shape? `print(batch.shape)`
- [ ] **Model Forward**: Does forward pass run without errors? `model(x)`
- [ ] **Loss Computation**: Is loss decreasing? `print(loss.item())`
- [ ] **Gradients**: Are gradients flowing? `print(param.grad is not None)`
- [ ] **Learning Rate**: Too high = NaN, too low = no learning
- [ ] **Optimizer State**: Are weights updating? `print(param.data)`
- [ ] **Device Placement**: CPU vs GPU? `print(x.device)`
- [ ] **Causal Mask**: Is mask applied correctly? Check attention scores
- [ ] **Shape Consistency**: Do all shapes match? `print(x.shape)`

## Debugging

Common issues:

1. **NaN values**: Check learning rate, gradient clipping
2. **Slow training**: Check batch size, use GPU
3. **Poor quality**: More training steps, better data
4. **OOM errors**: Reduce batch size, use gradient checkpointing

---

## Complete Layer-by-Layer Breakdown

> This section provides a detailed breakdown of every single layer in Thinker, combining **deep theoretical understanding** with **practical implementation details**. All explanations are **strictly based on our actual code** in `omni/thinker.py` and `omni/utils.py`.

### Theoretical Foundation: Why This Architecture?

#### The Decoder-Only Design

Thinker uses a **decoder-only** architecture (like GPT), not encoder-decoder (like BERT):

**Decoder-only advantages**:
- **Autoregressive generation**: Can generate text token by token
- **Unified architecture**: Same model for understanding and generation
- **Causal attention**: Only sees previous tokens (realistic for generation)
- **Simplicity**: Single stack of layers, easier to train

**Why not encoder-decoder?**
- More complex (two stacks)
- Encoder sees all tokens (not realistic for generation)
- Requires separate training for understanding vs generation

#### Pre-Normalization Architecture

Thinker uses **pre-norm** (normalize before transformation), not post-norm:

**Pre-norm benefits**:
- **Stable gradients**: Normalized inputs prevent gradient explosion
- **Deep networks**: Enables training very deep models (100+ layers)
- **Modern standard**: Used in LLaMA, PaLM, GPT-3

**Post-norm issues**:
- Can have gradient problems in deep networks
- Less stable training
- Older architecture (original Transformer)

### Why Pre-Norm Instead of Post-Norm?

#### Post-Norm (Original Transformer)
```python
# Normalize AFTER transformation
x = x + attention(x)
x = norm(x)  # Normalize AFTER
```
**Problem**: Gradient issues in deep networks (gradients can explode/vanishing)

#### Pre-Norm (μOmni)
```python
# Normalize BEFORE transformation
x = x + attention(norm(x))  # Normalize BEFORE
```
**Benefit**: Stable gradients, enables 100+ layers without gradient problems

**Key Insight**: Pre-norm normalizes the input to attention/MLP, making the transformation more stable. This is why modern models (LLaMA, GPT-3) use pre-norm.

#### Why These Specific Dimensions?

**d_model = 256**:
- Balance between capacity and efficiency
- Large enough for rich representations
- Small enough for 12GB GPU constraint

**n_heads = 4**:
- Each head gets 64 dimensions (256/4)
- Enough heads for specialization
- Not too many (diminishing returns)

**d_ff = 1024**:
- 4× expansion (standard ratio)
- Provides processing capacity
- Not excessive (efficiency)

### Complete Architecture Flow

```
Input: Token IDs [1, 1234, 5678]
    ↓
[Token Embedding] → (B, T, 256)
    ↓
┌─────────────────────────────────────┐
│  Transformer Block 1               │
│  ┌───────────────────────────────┐ │
│  │ RMSNorm                      │ │
│  └───────────┬───────────────────┘ │
│              ↓                      │
│  ┌───────────────────────────────┐ │
│  │ Multi-Head Attention          │ │
│  │  - Q/K/V Projections          │ │
│  │  - RoPE (Rotary Position)     │ │
│  │  - Attention Computation      │ │
│  │  - Output Projection           │ │
│  └───────────┬───────────────────┘ │
│              ↓                      │
│  [Residual: x = x + attn]           │
│              ↓                      │
│  ┌───────────────────────────────┐ │
│  │ RMSNorm                      │ │
│  └───────────┬───────────────────┘ │
│              ↓                      │
│  ┌───────────────────────────────┐ │
│  │ MLP (SwiGLU)                  │ │
│  │  - Gate Projection            │ │
│  │  - Up Projection              │ │
│  │  - Swish Activation           │ │
│  │  - Down Projection             │ │
│  └───────────┬───────────────────┘ │
│              ↓                      │
│  [Residual: x = x + mlp]            │
└───────────────┬──────────────────────┘
                ↓
    [Repeat for N blocks (default: 4)]
                ↓
[Final RMSNorm] → (B, T, 256)
    ↓
[Output Head] → (B, T, 5000)
    ↓
Token Logits
```

### Layer 1: Token Embedding

#### Code Location
`omni/thinker.py` - `ThinkerLM.__init__` and `forward`

#### Implementation

```python
class ThinkerLM(nn.Module):
    def __init__(self, vocab_size, ...):
        # Token embedding layer
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        # vocab_size = 5000 (number of unique tokens)
        # d_model = 256 (embedding dimension)
```

#### Forward Pass

```python
def forward(self, x, embeddings=None):
    if embeddings is None:
        # Convert token IDs to embeddings
        x = self.tok_emb(x)  # (B, T) → (B, T, d_model)
    else:
        # Use provided embeddings (for multimodal)
        x = embeddings
```

#### Deep Theoretical Analysis

**Why Embedding Dimension Matters**

**256 dimensions** is a carefully chosen balance:

**Too small (64-128)**:
- Insufficient capacity for rich representations
- May lose semantic nuances
- Faster but less capable

**Too large (512-1024)**:
- More parameters (memory intensive)
- Risk of overfitting
- Diminishing returns

**256 dimensions**:
- Sweet spot for small models
- Enough capacity for good representations
- Efficient for 12GB GPU constraint

**What Gets Learned**

During training, embeddings learn:
1. **Frequency patterns**: Common words get stable embeddings
2. **Syntactic patterns**: Similar grammatical roles cluster
3. **Semantic patterns**: Similar meanings cluster
4. **Context patterns**: Words used in similar contexts cluster

**What Value Do We Get?**

1. **Semantic Understanding**: Model understands word meanings
2. **Efficient Storage**: Dense vectors vs sparse one-hot
3. **Transfer Learning**: Embeddings work across tasks
4. **Generalization**: Captures patterns beyond training
5. **Interpretability**: Can visualize and analyze embedding space

### Layer 2: Transformer Blocks

#### Block Structure

Each block contains:
1. Pre-norm RMSNorm
2. Multi-Head Attention
3. Residual connection
4. Pre-norm RMSNorm
5. MLP (SwiGLU)
6. Residual connection

#### Block 1: RMSNorm (Pre-Attention)

```python
# From omni/utils.py
class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))
        self.eps = 1e-6
    
    def forward(self, x):
        # x shape: (B, T, d_model)
        # Compute RMS: sqrt(mean(x²))
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize: x / rms
        normalized = x / rms
        # Scale: normalized * scale
        return normalized * self.scale
```

**Why Pre-Norm Before Attention?**

**Pre-norm** (used in Thinker): `x = x + attention(norm(x))`
- Normalizes **before** attention
- Attention receives normalized inputs
- More stable gradients

**Post-norm** (original): `x = norm(x + attention(x))`
- Normalizes **after** attention
- Attention receives unnormalized inputs
- Can have gradient issues

**What Value Do We Get from RMSNorm?**

1. **Stable Training**: Prevents activation explosion
2. **Faster Convergence**: Enables larger learning rates
3. **Deep Networks**: Enables training 100+ layer models
4. **Efficiency**: Simpler than LayerNorm (no mean)
5. **Flexibility**: Learnable scale provides expressiveness

#### Block 2: Multi-Head Attention

**Q/K/V Projections**

```python
# From omni/thinker.py - Attention class
class Attention(nn.Module):
    def __init__(self, d, heads, ...):
        if use_gqa:
            # GQA: separate Q, K, V projections
            self.q = nn.Linear(d, heads * dk, bias=False)
            self.k = nn.Linear(d, kv_groups * dk, bias=False)
            self.v = nn.Linear(d, kv_groups * dk, bias=False)
        else:
            # Standard: combined QKV projection
            self.qkv = nn.Linear(d, 3*d, bias=False)
```

**Apply RoPE (Rotary Position Embedding)**

```python
# From omni/utils.py
class RoPE:
    def forward(self, q, k, pos):
        # Rotate query/key vectors based on position
        # q,k: (B, H, T, D), pos: (T,)
        # ... rotation computation ...
        return q_rotated, k_rotated
```

**Why RoPE in Attention?**

**RoPE is applied to Q and K** (not V):
- **Query and Key**: Used for computing similarity (need position info)
- **Value**: Contains actual information (position less critical)

**Why rotate instead of add?**
- **Additive** (learned embeddings): Fixed absolute positions
- **Rotative** (RoPE): Relative positions encoded in angles
- **Benefit**: Generalizes to longer sequences

**Attention Computation**

```python
# Compute attention scores
scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.dk)

# Apply causal mask (for decoder)
if causal:
    mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1)
    scores = scores.masked_fill(mask.bool(), float('-inf'))

# Softmax to get attention weights
attn_weights = torch.softmax(scores, dim=-1)  # (B, heads, T, T)

# Weighted sum of values
attn_output = torch.matmul(attn_weights, V)
```

**Why Scale by √dk?**

**The problem**: Dot products grow with dimension
- High dimensions → large dot products
- Large values → extreme softmax (nearly one-hot)
- Extreme softmax → vanishing gradients

**The solution**: Scale by `√dk`
- Normalizes variance of dot products
- Keeps softmax in "soft" region
- Maintains gradient flow

**What Value Do We Get from Attention?**

1. **Contextual Understanding**: Each token sees full previous context
2. **Flexible Relationships**: Learns any relationship type
3. **Parallel Processing**: All positions processed simultaneously
4. **Interpretability**: Attention weights show focus
5. **Efficiency**: Single matrix operation for all relationships

#### Block 3: MLP (SwiGLU)

**Gate and Up Projections**

```python
# From omni/thinker.py - MLP class
class MLP(nn.Module):
    def __init__(self, d, ff, use_swiglu=True):
        if use_swiglu:
            self.gate_proj = nn.Linear(d, ff, bias=False)  # Gate projection
            self.up_proj = nn.Linear(d, ff, bias=False)    # Up projection
            self.down_proj = nn.Linear(ff, d, bias=False) # Down projection
```

**Forward Pass**

```python
def forward(self, x):
    # x shape: (B, T, d_model)
    
    # Project to gate and up
    gate = self.gate_proj(x)  # (B, T, d_model) → (B, T, ff)
    up = self.up_proj(x)      # (B, T, d_model) → (B, T, ff)
    
    # Swish activation: x * sigmoid(x)
    swish = gate * torch.sigmoid(gate)  # (B, T, ff)
    
    # Element-wise multiplication
    x = swish * up  # (B, T, ff)
    
    # Project back to model dimension
    x = self.down_proj(x)  # (B, T, ff) → (B, T, d_model)
    
    return x
```

**Why SwiGLU?**

**Traditional MLP** (GELU):
- Single projection path
- All information processed equally
- Less expressive

**SwiGLU** (Swish-Gated Linear Unit):
- Two projections (gate + up)
- Gate learns "what to process"
- Up learns "how to process"
- Multiplication creates interaction

**Why this is better**:
- **Selective processing**: Gate filters information
- **More parameters**: Two projections vs one
- **Interaction**: Gate and up interact multiplicatively
- **Proven**: Used in PaLM, LLaMA (state-of-the-art)

**What Value Do We Get from SwiGLU?**

1. **Selective Processing**: Learns what information to use
2. **Smooth Gradients**: No dead neurons, better training
3. **Expressiveness**: More powerful than GELU/ReLU
4. **Efficiency**: Self-gating is parameter-efficient
5. **Proven**: Used in state-of-the-art models

### Layer 3: Final Normalization

```python
# After all blocks
x = self.norm(x)  # Final RMSNorm
# Shape: (B, T, d_model)
```

### Layer 4: Output Head

```python
# Convert hidden states to token predictions
self.head = nn.Linear(d_model, vocab_size)

# Forward
logits = self.head(x)  # (B, T, d_model) → (B, T, vocab_size)
# logits[0, 0, :] = scores for all 5000 tokens at position 0
```

**What are Logits?**

```python
# Logits are raw scores (before softmax)
logits = head(hidden_states)  # (1, 3, 5000)

# Convert to probabilities
probs = torch.softmax(logits, dim=-1)  # (1, 3, 5000)
# probs[0, 0, token_id] = probability of that token

# Get most likely token
next_token = torch.argmax(logits[0, -1, :])  # Token ID
```

**Why Linear Head?**

**Why not more layers?**
- **Single linear**: Sufficient for token prediction
- **More layers**: Overkill, risk of overfitting
- **Efficiency**: Faster computation

**Why not non-linearity?**
- **Softmax provides non-linearity**: Converts scores to probabilities
- **Additional non-linearity**: Redundant, may hurt
- **Standard practice**: Linear head + softmax is standard

**What Value Do We Get from Output Head?**

1. **Token Prediction**: Converts hidden states to token scores
2. **Efficiency**: Simple linear layer (fast)
3. **Flexibility**: Can predict any vocabulary token
4. **Interpretability**: Logits show model confidence
5. **Standard**: Matches standard transformer architecture

### Complete Forward Pass with Shapes

```python
# Input
token_ids = torch.tensor([[1, 1234, 5678]])  # (1, 3)

# Step 1: Embedding
x = tok_emb(token_ids)  # (1, 3, 256)

# Step 2: Block 1
x = norm1(x)           # (1, 3, 256)
x = attention(x)       # (1, 3, 256)
x = x + residual       # (1, 3, 256)
x = norm2(x)           # (1, 3, 256)
x = mlp(x)             # (1, 3, 256)
x = x + residual       # (1, 3, 256)

# Step 3-5: Blocks 2-4 (same process)
x = block2(x)  # (1, 3, 256)
x = block3(x)  # (1, 3, 256)
x = block4(x)  # (1, 3, 256)

# Step 6: Final norm
x = final_norm(x)  # (1, 3, 256)

# Step 7: Output head
logits = head(x)  # (1, 3, 5000)
```

### Memory and Computation

**Per Block**:
- **Parameters**: ~1M (d_model=256, n_heads=4, d_ff=1024)
- **Memory**: O(B * T * d_model) for activations
- **Computation**: O(B * T² * d_model) for attention

**Total Model**:
- **Parameters**: ~4M (4 blocks) + embeddings + head
- **Total**: ~50M parameters

## 📊 Component Comparison

| Component | Input Shape | Output Shape | Parameters | Purpose |
|-----------|-------------|--------------|------------|---------|
| Token Embedding | (B, T) | (B, T, 256) | ~1.3M | Token IDs → Embeddings |
| Transformer Block | (B, T, 256) | (B, T, 256) | ~1M | Process embeddings |
| Output Head | (B, T, 256) | (B, T, 5000) | ~1.3M | Embeddings → Token logits |
| **Total Thinker** | **(B, T)** | **(B, T, 5000)** | **~50M** | **Complete LLM** |

## ✅ Understanding Checkpoint

Before moving on, can you answer:

1. **Why does Thinker use causal masking?**
   - Answer: To prevent seeing future tokens during training (realistic for generation)

2. **What's the difference between RoPE and learned position embeddings?**
   - Answer: RoPE rotates vectors by position angle (relative), learned embeddings add absolute position vectors

3. **How does SwiGLU differ from GELU?**
   - Answer: SwiGLU uses gate mechanism (two projections, multiplicative), GELU is single projection with activation

4. **Why do we need projectors for different modalities?**
   - Answer: To align different embedding dimensions (vision=128, audio=192) to Thinker's dimension (256)

5. **What's the benefit of pre-norm over post-norm?**
   - Answer: Stable gradients, enables training very deep networks (100+ layers)

## 📖 Related Papers

- **Attention Is All You Need** (Vaswani et al., 2017)
  - Original transformer architecture
  - See: Section on Multi-Head Attention
  
- **LLaMA** (Touvron et al., 2023)
  - Pre-norm, RoPE, SwiGLU (same as μOmni)
  - See: Architecture section
  
- **RoFormer** (Su et al., 2021)
  - Rotary Position Embedding (RoPE)
  - See: RoPE implementation details

## 🚀 Extension Ideas

Want to experiment? Try these:

1. **Add MoE**: Implement mixture of experts for larger capacity
2. **Custom Activation**: Try different activations (GELU, ReLU, etc.)
3. **Layer Scaling**: Add learnable scaling factors to residual connections
4. **Flash Attention**: Implement memory-efficient attention
5. **Quantization**: Add INT8 quantization for faster inference

## ❓ Frequently Asked Questions

**Q: Why is Thinker decoder-only, not encoder-decoder?**
A: Decoder-only enables autoregressive generation (like GPT). Encoder-decoder is for tasks like translation.

**Q: Can I use Thinker without other components?**
A: Yes! Thinker can work standalone for text-only tasks.

**Q: Why 4 layers? Can I add more?**
A: 4 layers fit in 12GB VRAM. You can add more, but need more memory.

**Q: What's the difference between Thinker and GPT?**
A: Thinker is designed for multimodal (text/image/audio), GPT is text-only. Architecture is similar.

**Q: How do I modify Thinker for my task?**
A: Change vocab_size, d_model, n_layers in config. For new tasks, modify output head.

---

**Next:**
- [04_Audio_Encoder.md](04_Audio_Encoder.md) - How audio is processed
- [05_Vision_Encoder.md](05_Vision_Encoder.md) - How images are processed
- [07_Training_Workflow.md](07_Training_Workflow.md) - How to train Thinker

**See Also:**
- [Architecture Overview](02_Architecture_Overview.md)
- [Inference Guide](08_Inference_Guide.md)
- [Transformer Deep Dive](10_Transformer_Deep_Dive.md) - General transformer theory

