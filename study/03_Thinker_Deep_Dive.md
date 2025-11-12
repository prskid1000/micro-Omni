# Thinker: The Core Language Model

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

### 2. Transformer Blocks

Each block contains:
- **RMSNorm**: Normalization layer
- **Self-Attention**: Looks at all positions
- **MLP**: Feedforward network

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

**Multi-Head Attention**: Run attention multiple times in parallel:

```python
# 4 heads, each with 64 dimensions (256 total)
head1 = attention(q1, k1, v1)  # 64 dims
head2 = attention(q2, k2, v2)  # 64 dims
head3 = attention(q3, k3, v3)  # 64 dims
head4 = attention(q4, k4, v4)  # 64 dims
# Concatenate: 256 dims
```

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

**Why RoPE?**
- Better generalization to longer sequences
- Relative position encoding
- More efficient than absolute positions

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

**Benefit**: Reduces memory usage while maintaining quality.

### MoE (Mixture of Experts)

Optional feature - replaces MLP with multiple "expert" networks:

```python
# Router selects top-2 experts per token
experts = [Expert1(), Expert2(), ..., Expert8()]
selected = router(x)  # Top 2 experts
output = selected[0](x) + selected[1](x)
```

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

## Performance Tips

1. **Batch Processing**: Process multiple sequences together
2. **KV Caching**: Cache attention states during generation
3. **Mixed Precision**: Use float16 for faster training
4. **Gradient Accumulation**: Simulate larger batches

## Debugging

Common issues:

1. **NaN values**: Check learning rate, gradient clipping
2. **Slow training**: Check batch size, use GPU
3. **Poor quality**: More training steps, better data
4. **OOM errors**: Reduce batch size, use gradient checkpointing

---

**Next:**
- [04_Audio_Encoder.md](04_Audio_Encoder.md) - How audio is processed
- [05_Vision_Encoder.md](05_Vision_Encoder.md) - How images are processed
- [07_Training_Workflow.md](07_Training_Workflow.md) - How to train Thinker

**See Also:**
- [Architecture Overview](02_Architecture_Overview.md)
- [Inference Guide](08_Inference_Guide.md)

