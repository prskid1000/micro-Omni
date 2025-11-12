# Thinker: Complete Layer-by-Layer Breakdown

## Overview

This document explains **every single layer** in Thinker, combining **deep theoretical understanding** with **practical implementation details**. We'll explore not just what each layer does, but **why** it's designed that way and **what value** it provides.

## Theoretical Foundation: Why This Architecture?

### The Decoder-Only Design

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

### Pre-Normalization Architecture

Thinker uses **pre-norm** (normalize before transformation), not post-norm:

**Pre-norm benefits**:
- **Stable gradients**: Normalized inputs prevent gradient explosion
- **Deep networks**: Enables training very deep models (100+ layers)
- **Modern standard**: Used in LLaMA, PaLM, GPT-3

**Post-norm issues**:
- Can have gradient problems in deep networks
- Less stable training
- Older architecture (original Transformer)

### Why These Specific Dimensions?

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

## Layer-by-Layer Deep Dive

## Complete Architecture

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

## Layer 1: Token Embedding

### Code Location
`omni/thinker.py` - `ThinkerLM.__init__` and `forward`

### Implementation

```python
class ThinkerLM(nn.Module):
    def __init__(self, vocab_size, ...):
        # Token embedding layer
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        # vocab_size = 5000 (number of unique tokens)
        # d_model = 256 (embedding dimension)
```

### Forward Pass

```python
def forward(self, x, embeddings=None):
    if embeddings is None:
        # Convert token IDs to embeddings
        x = self.tok_emb(x)  # (B, T) → (B, T, d_model)
    else:
        # Use provided embeddings (for multimodal)
        x = embeddings
```

### What Happens

1. **Input**: Token IDs `[1, 1234, 5678]` shape `(1, 3)`
2. **Lookup**: Each ID maps to a 256-dim vector
3. **Output**: Embeddings shape `(1, 3, 256)`

### Deep Theoretical Analysis

#### Why Embedding Dimension Matters

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

#### Embedding Table Structure

The embedding table is a **learned lookup**:
- **Rows**: One per vocabulary token (5000 rows)
- **Columns**: Embedding dimensions (256 columns)
- **Total**: 5000 × 256 = 1.28M parameters

**Why this size?**
- Vocabulary size (5000) balances coverage vs efficiency
- Too small: Out-of-vocabulary issues
- Too large: Wasted parameters on rare tokens

#### What Gets Learned

During training, embeddings learn:
1. **Frequency patterns**: Common words get stable embeddings
2. **Syntactic patterns**: Similar grammatical roles cluster
3. **Semantic patterns**: Similar meanings cluster
4. **Context patterns**: Words used in similar contexts cluster

#### Information Content

Each 256-dim embedding encodes:
- **Semantic meaning**: What the word means
- **Syntactic role**: How it's used grammatically
- **Contextual usage**: Where it appears
- **Relationships**: How it relates to other words

This is a **compressed representation** of all the information needed about each token.

#### What Value Do We Get?

1. **Semantic Understanding**: Model understands word meanings
2. **Efficient Storage**: Dense vectors vs sparse one-hot
3. **Transfer Learning**: Embeddings work across tasks
4. **Generalization**: Captures patterns beyond training
5. **Interpretability**: Can visualize and analyze embedding space

### Example

```python
# Input
token_ids = torch.tensor([[1, 1234, 5678]])  # (1, 3)

# Embedding table (simplified)
# embedding_table shape: (5000, 256)
# embedding_table[1] = [0.1, -0.2, 0.3, ...]  # 256 numbers
# embedding_table[1234] = [0.5, 0.1, -0.4, ...]
# embedding_table[5678] = [-0.1, 0.6, 0.2, ...]

# Output
embeddings = tok_emb(token_ids)
# Shape: (1, 3, 256)
# embeddings[0, 0, :] = embedding_table[1]
# embeddings[0, 1, :] = embedding_table[1234]
# embeddings[0, 2, :] = embedding_table[5678]
```

## Layer 2: Transformer Blocks

### Block Structure

Each block contains:
1. Pre-norm RMSNorm
2. Multi-Head Attention
3. Residual connection
4. Pre-norm RMSNorm
5. MLP (SwiGLU)
6. Residual connection

### Block 1: RMSNorm (Pre-Attention)

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

### Step-by-Step

```python
# Input to norm
x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])  # (1, 1, 4)

# Step 1: Square
x_squared = x ** 2  # [[[1, 4, 9, 16]]]

# Step 2: Mean along last dimension
mean_sq = x_squared.mean(dim=-1, keepdim=True)  # [[[7.5]]]

# Step 3: RMS
rms = sqrt(7.5 + 1e-6)  # ~2.74

# Step 4: Normalize
normalized = x / rms  # [[[0.365, 0.730, 1.095, 1.460]]]

# Step 5: Scale (if scale = [1, 1, 1, 1])
output = normalized * scale  # Same
```

### Why Pre-Norm Before Attention?

**The normalization order matters**:

**Pre-norm** (used in Thinker): `x = x + attention(norm(x))`
- Normalizes **before** attention
- Attention receives normalized inputs
- More stable gradients

**Post-norm** (original): `x = norm(x + attention(x))`
- Normalizes **after** attention
- Attention receives unnormalized inputs
- Can have gradient issues

**Why pre-norm is better**:
- Attention weights are more stable
- Gradients flow better
- Enables deeper networks
- Modern standard (LLaMA, PaLM use pre-norm)

### What Value Do We Get from RMSNorm?

1. **Stable Training**: Prevents activation explosion
2. **Faster Convergence**: Enables larger learning rates
3. **Deep Networks**: Enables training 100+ layer models
4. **Efficiency**: Simpler than LayerNorm (no mean)
5. **Flexibility**: Learnable scale provides expressiveness

### Block 2: Multi-Head Attention

#### 2.1: Q/K/V Projections

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

#### Forward Pass - Projections

```python
def forward(self, x, ...):
    B, T, D = x.shape  # (batch, seq_len, d_model)
    
    if self.use_gqa:
        # Separate projections
        Q = self.q(x)  # (B, T, heads * dk)
        K = self.k(x)  # (B, T, kv_groups * dk)
        V = self.v(x)  # (B, T, kv_groups * dk)
    else:
        # Combined projection, then split
        qkv = self.qkv(x)  # (B, T, 3*d)
        Q, K, V = qkv.chunk(3, dim=-1)  # Each: (B, T, d)
```

#### 2.2: Reshape for Multi-Head

```python
# Reshape to separate heads
# Q: (B, T, heads * dk) → (B, heads, T, dk)
Q = Q.view(B, T, self.h, self.dk).transpose(1, 2)
K = K.view(B, T, self.kv_groups, self.dk).transpose(1, 2)
V = V.view(B, T, self.kv_groups, self.dk).transpose(1, 2)

# Example with 4 heads, dk=64
# Before: (1, 3, 256)
# After:  (1, 4, 3, 64)
# - 1 batch
# - 4 heads
# - 3 sequence positions
# - 64 dimensions per head
```

#### 2.3: Apply RoPE (Rotary Position Embedding)

```python
# From omni/utils.py
class RoPE:
    def __init__(self, dim, theta=10000.0):
        # Create frequency matrix
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2) / dim))
        self.register_buffer('inv_freq', inv_freq)
    
    def forward(self, x, positions):
        # x: (B, heads, T, dk)
        # positions: (T,) - [0, 1, 2, ...]
        
        # Compute angles
        angles = positions.unsqueeze(-1) * self.inv_freq  # (T, dk/2)
        
        # Split into real/imaginary parts
        x_real = x[..., 0::2]  # Even indices
        x_imag = x[..., 1::2]  # Odd indices
        
        # Rotate
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        
        x_rot_real = x_real * cos - x_imag * sin
        x_rot_imag = x_real * sin + x_imag * cos
        
        # Interleave back
        x_rot = torch.zeros_like(x)
        x_rot[..., 0::2] = x_rot_real
        x_rot[..., 1::2] = x_rot_imag
        
        return x_rot
```

### Deep Dive: Why RoPE in Attention?

**RoPE is applied to Q and K** (not V):
- **Query and Key**: Used for computing similarity (need position info)
- **Value**: Contains actual information (position less critical)

**Why rotate instead of add?**
- **Additive** (learned embeddings): Fixed absolute positions
- **Rotative** (RoPE): Relative positions encoded in angles
- **Benefit**: Generalizes to longer sequences

**Mathematical elegance**:
- Rotation preserves vector magnitude
- Only direction changes (encodes position)
- Relative positions encoded in angle differences

### What Value Do We Get from RoPE?

1. **Generalization**: Works on sequences longer than training
2. **Relative Understanding**: Learns "distance" not "absolute position"
3. **Efficiency**: No learnable parameters (computed on-the-fly)
4. **Multi-scale**: Different frequencies capture different scales
5. **Proven**: Used in state-of-the-art models (LLaMA, PaLM)

#### 2.4: Attention Computation

```python
# Compute attention scores
# Q @ K^T: (B, heads, T, dk) @ (B, heads, dk, T) → (B, heads, T, T)
scores = torch.matmul(Q, K.transpose(-2, -1))

# Scale by sqrt(dk)
scores = scores / math.sqrt(self.dk)

# Apply causal mask (for decoder)
if causal:
    mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1)
    scores = scores.masked_fill(mask.bool(), float('-inf'))

# Softmax to get attention weights
attn_weights = torch.softmax(scores, dim=-1)  # (B, heads, T, T)
# Each row sums to 1.0

# Apply dropout
attn_weights = self.drop(attn_weights)

# Weighted sum of values
# attn_weights @ V: (B, heads, T, T) @ (B, heads, T, dk) → (B, heads, T, dk)
attn_output = torch.matmul(attn_weights, V)
```

### Deep Dive: Causal Masking

**Why causal mask?**
- **Decoder-only**: Should only see previous tokens (realistic generation)
- **Training**: Prevents cheating (can't see future tokens)
- **Inference**: Matches training conditions

**How it works**:
- Upper triangle set to `-inf`
- After softmax: `-inf` → 0 probability
- Result: Position i can only attend to positions ≤ i

**Mathematical view**:
```
Without mask: All positions can attend to all positions
With mask: Position i can only attend to positions j where j ≤ i
```

### Why Scale by √dk?

**The problem**: Dot products grow with dimension
- High dimensions → large dot products
- Large values → extreme softmax (nearly one-hot)
- Extreme softmax → vanishing gradients

**The solution**: Scale by `√dk`
- Normalizes variance of dot products
- Keeps softmax in "soft" region
- Maintains gradient flow

**Mathematical justification**:
- Variance of Q·K scales with `dk`
- Scaling by `√dk` normalizes to unit variance
- Keeps attention weights distributed (not peaked)

### What Value Do We Get from Attention?

1. **Contextual Understanding**: Each token sees full previous context
2. **Flexible Relationships**: Learns any relationship type
3. **Parallel Processing**: All positions processed simultaneously
4. **Interpretability**: Attention weights show focus
5. **Efficiency**: Single matrix operation for all relationships

#### 2.5: Concatenate Heads

```python
# Reshape back: (B, heads, T, dk) → (B, T, heads * dk)
attn_output = attn_output.transpose(1, 2).contiguous()
attn_output = attn_output.view(B, T, self.h * self.dk)
# Result: (B, T, d_model)
```

#### 2.6: Output Projection

```python
# Final linear projection
output = self.o(attn_output)  # (B, T, d_model) → (B, T, d_model)
```

### Block 3: Residual Connection

```python
# After attention
residual = x  # Save input
x = self.norm1(x)  # Normalize
x, kv_cache = self.attention(x, ...)  # Attention
x = x + residual  # Add residual
# Shape: (B, T, d_model)
```

### Block 4: MLP (SwiGLU)

#### 4.1: Gate and Up Projections

```python
# From omni/thinker.py - MLP class
class MLP(nn.Module):
    def __init__(self, d, ff, use_swiglu=True):
        if use_swiglu:
            self.gate_proj = nn.Linear(d, ff, bias=False)  # Gate projection
            self.up_proj = nn.Linear(d, ff, bias=False)    # Up projection
            self.down_proj = nn.Linear(ff, d, bias=False) # Down projection
```

#### 4.2: Forward Pass

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

### Deep Dive: Why SwiGLU?

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

### The Gating Mechanism

**Gate function**: `swish(gate) = gate × sigmoid(gate)`
- When `gate > 0`: High value (information flows)
- When `gate < 0`: Low value (information blocked)
- Smooth transition (unlike ReLU's hard cutoff)

**Why smooth?**
- Differentiable everywhere
- No dead neurons
- Better gradient flow

### What Value Do We Get from SwiGLU?

1. **Selective Processing**: Learns what information to use
2. **Smooth Gradients**: No dead neurons, better training
3. **Expressiveness**: More powerful than GELU/ReLU
4. **Efficiency**: Self-gating is parameter-efficient
5. **Proven**: Used in state-of-the-art models

#### Step-by-Step Example

```python
# Input
x = torch.randn(1, 3, 256)  # (B, T, d_model)

# Gate projection
gate = gate_proj(x)  # (1, 3, 1024)

# Up projection
up = up_proj(x)  # (1, 3, 1024)

# Swish: gate * sigmoid(gate)
swish = gate * torch.sigmoid(gate)  # (1, 3, 1024)

# Multiply
combined = swish * up  # (1, 3, 1024)

# Down projection
output = down_proj(combined)  # (1, 3, 256)
```

### Block 5: Final Residual

```python
# After MLP
residual = x  # Save input
x = self.norm2(x)  # Normalize
x = self.mlp(x)    # MLP
x = x + residual   # Add residual
```

## Layer 3: Final Normalization

```python
# After all blocks
x = self.norm(x)  # Final RMSNorm
# Shape: (B, T, d_model)
```

## Layer 4: Output Head

```python
# Convert hidden states to token predictions
self.head = nn.Linear(d_model, vocab_size)

# Forward
logits = self.head(x)  # (B, T, d_model) → (B, T, vocab_size)
# logits[0, 0, :] = scores for all 5000 tokens at position 0
```

### What are Logits?

```python
# Logits are raw scores (before softmax)
logits = head(hidden_states)  # (1, 3, 5000)

# Convert to probabilities
probs = torch.softmax(logits, dim=-1)  # (1, 3, 5000)
# probs[0, 0, token_id] = probability of that token

# Get most likely token
next_token = torch.argmax(logits[0, -1, :])  # Token ID
```

### Deep Dive: Why Linear Head?

**Why not more layers?**
- **Single linear**: Sufficient for token prediction
- **More layers**: Overkill, risk of overfitting
- **Efficiency**: Faster computation

**Why not non-linearity?**
- **Softmax provides non-linearity**: Converts scores to probabilities
- **Additional non-linearity**: Redundant, may hurt
- **Standard practice**: Linear head + softmax is standard

### Logits vs Probabilities

**Logits** (raw scores):
- Can be any real number
- Not normalized
- Efficient to compute

**Probabilities** (after softmax):
- Sum to 1.0
- All values positive
- Interpretable as probabilities

**Why use logits during training?**
- **Numerical stability**: Softmax in loss function (more stable)
- **Efficiency**: Don't need probabilities for loss
- **Standard practice**: Cross-entropy expects logits

### What Value Do We Get from Output Head?

1. **Token Prediction**: Converts hidden states to token scores
2. **Efficiency**: Simple linear layer (fast)
3. **Flexibility**: Can predict any vocabulary token
4. **Interpretability**: Logits show model confidence
5. **Standard**: Matches standard transformer architecture

## Complete Forward Pass with Shapes

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

# Step 3: Block 2 (same process)
x = block2(x)  # (1, 3, 256)

# Step 4: Block 3
x = block3(x)  # (1, 3, 256)

# Step 5: Block 4
x = block4(x)  # (1, 3, 256)

# Step 6: Final norm
x = final_norm(x)  # (1, 3, 256)

# Step 7: Output head
logits = head(x)  # (1, 3, 5000)
```

## Key Parameters

From `configs/thinker_tiny.json`:

```json
{
  "vocab_size": 5000,    // Token embedding table size
  "d_model": 256,        // Embedding dimension
  "n_layers": 4,         // Number of transformer blocks
  "n_heads": 4,          // Attention heads
  "d_ff": 1024,          // MLP feedforward dimension
  "dropout": 0.1,        // Dropout rate
  "use_swiglu": true,    // Use SwiGLU activation
  "use_gqa": false       // Use grouped query attention
}
```

## Memory and Computation

### Per Block:
- **Parameters**: ~1M (d_model=256, n_heads=4, d_ff=1024)
- **Memory**: O(B * T * d_model) for activations
- **Computation**: O(B * T² * d_model) for attention

### Total Model:
- **Parameters**: ~4M (4 blocks) + embeddings + head
- **Total**: ~50M parameters

---

**Next:**
- [10_Transformer_Deep_Dive.md](10_Transformer_Deep_Dive.md) - General transformer theory
- [04_Audio_Encoder.md](04_Audio_Encoder.md) - Audio encoder layers
- [05_Vision_Encoder.md](05_Vision_Encoder.md) - Vision encoder layers

