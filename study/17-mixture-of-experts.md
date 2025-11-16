# Chapter 17: Mixture of Experts (MoE)

[← Previous: SwiGLU](16-swiglu-activation.md) | [Back to Index](00-INDEX.md) | [Next: Normalization →](18-normalization.md)

---

## 🎯 Learning Objectives

By the end of this chapter, you will understand:
- The scalability challenge of large models
- How Mixture of Experts (MoE) works
- Sparse vs dense computation
- Router networks and expert selection
- Trade-offs and when to use MoE
- How μOmni implements MoE

---

## ❓ The Scalability Problem

### Why Do We Need MoE?

**Analogy: Hospital Specialists**

```
PROBLEM: One Doctor for Everything (Dense FFN)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Imagine a hospital with ONE doctor who must:
- Treat heart problems
- Perform brain surgery
- Fix broken bones
- Deliver babies
- Treat infections
...

Problems:
❌ Doctor is overwhelmed
❌ Can't be expert at everything
❌ Slow (one patient at a time)
❌ Quality suffers

SOLUTION: Team of Specialists (MoE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Hospital with multiple specialists:
- Cardiologist (heart expert)
- Neurologist (brain expert)
- Orthopedist (bones expert)
- Obstetrician (delivery expert)
- Infectious disease specialist
...

+ Triage nurse (router!) who decides:
  "This patient needs the cardiologist and orthopedist"

Benefits:
✅ Each specialist is an expert in their domain
✅ Only activate needed specialists (efficient!)
✅ Higher total capacity (more specialists)
✅ Better quality (specialized knowledge)

This is EXACTLY how Mixture of Experts works!
```

**The Technical Problem:**

```
Traditional Dense FFN:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Every token goes through the SAME large network:

Token: "cat" → FFN (1024 neurons)
Token: "quantum" → FFN (1024 neurons)
Token: "plays" → FFN (1024 neurons)

Problems:
❌ Same network for all types of tokens
❌ Hard to specialize for different patterns
❌ Computation grows linearly with model size

To scale to GPT-4 size (1.8 trillion parameters):
- Need HUGE FFN layers
- Every token uses every parameter
- Compute cost: MASSIVE 💸💸💸

Mixture of Experts (MoE):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Multiple specialized networks (experts):

Token: "cat" → Router → Expert 2, Expert 5 (animal experts)
Token: "quantum" → Router → Expert 1, Expert 7 (science experts)
Token: "plays" → Router → Expert 3, Expert 6 (action experts)

Benefits:
✅ Specialized experts for different patterns
✅ Only activate 2-4 experts per token (sparse!)
✅ Total capacity: 8 experts = 8x parameters
✅ Compute cost: Only use 2 experts = 1/4 cost!

Scale to GPT-4:
- 8x more parameters for same compute!
- Each token only uses 2 experts
- Much more efficient 🚀
```

---

## 🏗️ How MoE Works: Step-by-Step

### The Complete Architecture

```
┌────────────────────────────────────────────┐
│         MIXTURE OF EXPERTS LAYER           │
│                                            │
│  Input token embedding: [0.2, 0.5, -0.3, ...]
│                ↓                            │
│  ┌─────────────────────────┐              │
│  │  ROUTER NETWORK         │              │
│  │  (Learned gating)       │              │
│  │                         │              │
│  │  W_router · input       │              │
│  │  → Softmax              │              │
│  └─────────────────────────┘              │
│                ↓                            │
│  Probabilities: [0.05, 0.12, 0.42, 0.03, 0.38, 0.08, 0.02, 0.01]
│                                            │
│  Top-2 experts:                            │
│  - Expert 2: 0.42 (42% weight)            │
│  - Expert 4: 0.38 (38% weight)            │
│                ↓                            │
│       ┌────────┴───────┐                  │
│       ↓                ↓                   │
│  ┌──────────┐    ┌──────────┐            │
│  │ Expert 2 │    │ Expert 4 │            │
│  │ (FFN)    │    │ (FFN)    │            │
│  └──────────┘    └──────────┘            │
│       ↓                ↓                   │
│  output_2        output_4                 │
│       ↓                ↓                   │
│  ┌─────────────────────────┐             │
│  │  WEIGHTED COMBINATION   │             │
│  │                         │             │
│  │  0.42 × output_2 +      │             │
│  │  0.38 × output_4        │             │
│  └─────────────────────────┘             │
│                ↓                           │
│         Final output                       │
└────────────────────────────────────────────┘
```

### Detailed Example

```
Input: Word "quantum"
Embedding: [0.5, -0.2, 0.8, ..., 0.3] (256-dim)

STEP 1: Router decides which experts to use
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Router network (learned weights):
  W_router · input → raw_scores
  → [2.1, 0.5, 1.8, 0.2, 0.9, 1.5, 0.3, 0.4]
  
Softmax → probabilities:
  → [0.28, 0.05, 0.21, 0.04, 0.08, 0.16, 0.05, 0.05]
  
Select top-2:
  Expert 0: 0.28 (28%) ← Science/technical expert
  Expert 2: 0.21 (21%) ← Abstract concepts expert
  
Normalize (sum to 1):
  Expert 0: 0.28 / (0.28+0.21) = 0.57
  Expert 2: 0.21 / (0.28+0.21) = 0.43

STEP 2: Process with selected experts
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Expert 0 (Science specialist):
  Input [0.5, -0.2, 0.8, ...]
  → FFN processing (1024 neurons)
  → Output [0.3, 0.7, -0.1, ...] (256-dim)

Expert 2 (Abstract concepts):
  Input [0.5, -0.2, 0.8, ...]
  → FFN processing (1024 neurons)
  → Output [0.2, 0.6, 0.1, ...] (256-dim)

STEP 3: Combine outputs
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Final = 0.57 × Expert_0_output + 0.43 × Expert_2_output

Final = 0.57 × [0.3, 0.7, -0.1, ...] + 
        0.43 × [0.2, 0.6, 0.1, ...]
      = [0.26, 0.66, 0.0, ...]

This final output goes to the next layer!
```

---

## 💡 Sparse Activation: The Key Insight

**Analogy: Library Sections**

```
DENSE (everyone searches entire library):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You want a book on physics:
- Search fiction section ❌ (waste of time)
- Search biography section ❌ (waste of time)
- Search history section ❌ (waste of time)
- Search science section ✓ (found it!)
- Continue searching other sections ❌ (waste of time)

Cost: Search ALL 8 sections = 8 units of work

SPARSE (MoE - only search relevant sections):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Router (librarian) says: "Try science and reference"
- Search science section ✓ (relevant!)
- Search reference section ✓ (relevant!)
- Skip other 6 sections ✓ (saved time!)

Cost: Search ONLY 2 sections = 2 units of work

Same result, 4x less work!
```

**Technical Benefits:**

```
Dense FFN (8192 neurons):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Every token → All 8192 neurons active
Computation per token: 8192 × 256 = 2.1M ops

MoE (8 experts, 1024 neurons each):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Every token → Only 2 experts × 1024 neurons
Computation per token: 2 × 1024 × 256 = 524K ops

Speed-up: 2.1M / 524K = 4x faster! 🚀

But total capacity:
Dense: 8192 neurons
MoE: 8 × 1024 = 8192 neurons (same!)

You get the SAME capacity for 1/4 the compute!
```

---

## 📊 Detailed Comparison

### Dense FFN vs MoE

```
DENSE FEEDFORWARD NETWORK:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Architecture:
  Input (256) → FFN (4096) → Output (256)

Parameters:
  W1: 256 × 4096 = 1.05M
  W2: 4096 × 256 = 1.05M
  Total: 2.1M parameters

Computation per token:
  Forward: 256 × 4096 + 4096 × 256 = 2.1M ops
  All neurons active always!

Capacity:
  One network learns everything
  Limited specialization

MIXTURE OF EXPERTS (8 experts, top-2):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Architecture:
  Input (256) → Router → Top-2 of 8 Experts → Output (256)
  Each expert: FFN (512 neurons)

Parameters:
  Router: 256 × 8 = 2K
  Expert 0-7: 8 × (256×512 + 512×256) = 2.1M each
  Total: 2K + 8 × 2.1M = 16.8M parameters (8x more!)

Computation per token:
  Router: 256 × 8 = 2K ops
  Expert selection: ~1K ops
  2 experts: 2 × (256×512 + 512×256) = 524K ops
  Total: ~527K ops (4x less than dense!)

Capacity:
  8 specialized experts
  High specialization
  But only 2 active per token (sparse!)

SUMMARY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
          │ Dense   │ MoE (8 experts, top-2)
─────────────────────────────────────────────────
Params    │ 2.1M    │ 16.8M (8x more)
Compute   │ 2.1M ops│ 527K ops (4x less)
Capacity  │ 4096    │ 8 × 512 = 4096 (same)
Active    │ 100%    │ 25% (sparse)
Specialize│ Low     │ High
```

---

## 🎯 The Router: Learning to Route

### How Does the Router Learn?

```
The router is a LEARNED network!
It's not hand-coded - it learns during training.

Training signal:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Router sends token to experts
2. Experts process token
3. Combined output makes prediction
4. Loss is computed (prediction vs ground truth)
5. Gradients flow back through:
   - Experts (how to process better)
   - Router (how to route better!)

The router learns patterns like:
- "Math tokens → Expert 1 and Expert 5"
- "Name tokens → Expert 2 and Expert 7"
- "Verb tokens → Expert 3 and Expert 6"

Over time, experts specialize!

Load Balancing:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem: What if all tokens go to Expert 0?
→ Expert 0 is overworked (bottleneck!)
→ Experts 1-7 are unused (wasted capacity!)

Solution: Add auxiliary loss
  Encourage balanced expert usage:
  - Penalize if one expert is used too much
  - Reward if all experts are used equally
  
Result: Even distribution of tokens across experts
```

### Router Implementation

```python
class Router(nn.Module):
    def __init__(self, d_model, num_experts):
        super().__init__()
        self.num_experts = num_experts
        self.gate = nn.Linear(d_model, num_experts)
    
    def forward(self, x, top_k=2):
        # x: (batch, seq_len, d_model)
        
        # Compute routing scores
        logits = self.gate(x)  # (batch, seq_len, num_experts)
        
        # Softmax to get probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Select top-k experts
        top_probs, top_indices = torch.topk(probs, top_k, dim=-1)
        # top_probs: (batch, seq_len, top_k)
        # top_indices: (batch, seq_len, top_k)
        
        # Normalize top-k probabilities to sum to 1
        top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)
        
        return top_indices, top_probs
```

---

## ⚖️ Trade-offs and Challenges

### Advantages

```
✅ SCALABILITY:
   Can scale to 100s of billions of parameters
   Without proportional compute increase
   
✅ SPECIALIZATION:
   Experts naturally learn different patterns
   Better representation of diverse data
   
✅ EFFICIENCY:
   4-8x less computation per token
   Enables larger models on same hardware
   
✅ MODULARITY:
   Can add/remove experts
   Can train experts separately
```

### Challenges

```
❌ TRAINING COMPLEXITY:
   Load balancing is tricky
   Need auxiliary losses
   Can have instability
   
❌ COMMUNICATION OVERHEAD:
   In distributed training, routing requires communication
   Can become bottleneck
   
❌ MEMORY:
   8x more parameters = 8x more memory
   All experts must fit in memory (even if inactive)
   
❌ INFERENCE:
   Need to load all experts
   Batch processing more complex
```

### When to Use MoE

```
USE MoE when:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Scaling to very large models (>10B parameters)
✅ Diverse data (benefits from specialization)
✅ Compute-constrained (need efficiency)
✅ Have infrastructure for distributed training

DON'T use MoE when:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❌ Small models (<1B parameters) - overhead not worth it
❌ Simple, homogeneous data - specialization not helpful
❌ Memory-constrained - dense is more memory efficient
❌ Single GPU training - communication overhead too high
```

---

## 🎯 μOmni's MoE Implementation

```python
# From omni/thinker.py (simplified)

class MoE(nn.Module):
    """Mixture of Experts layer (optional)"""
    def __init__(self, d, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Router
        self.router = nn.Linear(d, num_experts, bias=False)
        
        # Experts (each is a small FFN)
        d_ffn = 4 * d // num_experts  # Smaller FFN per expert
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, d_ffn, bias=False),
                nn.GELU(),
                nn.Linear(d_ffn, d, bias=False)
            )
            for _ in range(num_experts)
        ])
    
    def forward(self, x):
        B, T, D = x.shape
        
        # Route: which experts for each token?
        router_logits = self.router(x)  # (B, T, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        top_probs, top_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)
        
        # Process with selected experts
        output = torch.zeros_like(x)
        for k in range(self.top_k):
            expert_idx = top_indices[:, :, k]  # (B, T)
            expert_prob = top_probs[:, :, k]    # (B, T)
            
            # Batch process by expert (for efficiency)
            for e in range(self.num_experts):
                mask = (expert_idx == e)  # Which tokens go to expert e?
                if mask.any():
                    expert_input = x[mask]
                    expert_output = self.experts[e](expert_input)
                    output[mask] += expert_output * expert_prob[mask].unsqueeze(-1)
        
        return output

# Enable MoE in config:
{
    "use_moe": true,
    "num_experts": 8,
    "top_k": 2,
    "d_model": 256
}
```

### μOmni MoE Status

```
Current status in μOmni:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Status: OPTIONAL, EXPERIMENTAL
Default: OFF (uses standard dense FFN)

Reasons:
1. μOmni is small (120-140M params)
   → MoE overhead not worth it at this scale
   
2. Educational focus
   → Dense FFN is simpler to understand
   
3. Single GPU training
   → MoE communication overhead not beneficial
   
4. Memory constraints (12GB GPU)
   → 8x parameters would exceed memory

When to enable:
- Experimenting with MoE concepts
- Scaling μOmni to larger sizes (>1B)
- Multi-GPU training setup
```

---

## 🌟 Real-World MoE Models

```
Famous models using MoE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Switch Transformer (Google, 2021):
- 1.6 trillion parameters
- 2048 experts per layer!
- Top-1 routing
- 7x faster training than dense

GLaM (Google, 2021):
- 1.2 trillion parameters
- 64 experts, top-2 routing
- Outperforms GPT-3 with 1/3 compute

GPT-4 (OpenAI, 2023) - rumored:
- Likely uses MoE
- 8-16 experts
- Enables massive scale

Mixtral 8x7B (Mistral AI, 2024):
- 8 experts, 7B parameters each
- Top-2 routing
- 47B total params, 13B active
- Open-source, SOTA performance
```

---

## 💡 Key Takeaways

✅ **MoE** = Multiple specialized expert networks + router  
✅ **Sparse activation**: Only k of N experts per token (e.g., 2 of 8)  
✅ **Efficiency**: 4-8x less compute for same capacity  
✅ **Scalability**: Enables trillion-parameter models  
✅ **Specialization**: Experts naturally learn different patterns  
✅ **Router learns** which experts to use for each token  
✅ **Trade-off**: More parameters but less compute per token  
✅ **Best for**: Very large models (>10B params)  
✅ **μOmni**: Optional/experimental (better suited for larger scale)

---

## 🎓 Self-Check Questions

1. What problem does MoE solve?
2. How does sparse activation work?
3. What does the router do?
4. Why can MoE have more parameters but less computation?
5. When should you use MoE vs dense FFN?

<details>
<summary>📝 Click to see answers</summary>

1. MoE solves the scalability problem - allows models to grow to trillions of parameters while keeping per-token computation manageable
2. Sparse activation means only a subset (e.g., 2 of 8) experts are active for each token, not all of them - saving 4-8x computation
3. The router is a learned network that decides which k experts should process each token based on the token's features
4. MoE has more total parameters (8 experts) but each token only uses a few (2 experts), so computation per token is much less than using all parameters
5. Use MoE for very large models (>10B params) with diverse data on distributed systems. Use dense FFN for smaller models, single GPU, or simpler/more homogeneous data
</details>

---

[Continue to Chapter 18: Normalization Techniques →](18-normalization.md)

**Chapter Progress:** μOmni Architecture ●●○○○○○ (2/7 complete)

---

