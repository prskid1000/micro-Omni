[← Previous: 09-efficient-attention](09-efficient-attention.md) | [Index](00-INDEX.md) | [Next: 11-vector-quantization →](11-vector-quantization.md)

# Chapter 10: Mixture of Experts

---

## Learning Objectives

By the end of this chapter, you will understand:
- Why scaling model size does not have to mean scaling inference cost
- How a router selects a subset of experts for each token
- The load balancing problem and its solution
- micro-Omni's sorted batched dispatch optimization

---

## The Scaling Dilemma

Bigger models produce better results. This has been one of the most consistent findings in deep learning. But bigger models also mean:

- More computation per token (slower inference)
- More memory for weights (larger GPU requirements)
- More energy per query (higher operating costs)

The dream: a model with the quality of a large model but the speed of a small one. Mixture of Experts (MoE) gets surprisingly close to this dream.

---

## The Core Insight: Conditional Computation

Not every input needs every parameter. When a hospital receives a patient, the cardiologist examines heart problems and the neurologist examines brain problems. The patient does not see every specialist -- only the relevant ones. But the hospital has the collective expertise of all its doctors.

MoE applies this principle to neural networks. Instead of one large feedforward network (FFN) that processes every token, you have N smaller expert FFNs. A lightweight router decides which experts each token should visit. Only a few experts activate per token.

```
STANDARD TRANSFORMER BLOCK:           MoE TRANSFORMER BLOCK:

Input -> Attention -> FFN -> Output    Input -> Attention -> Router -> Output
                      |                                       |
                  [one big FFN]                    +--->  Expert 1
                  [processes every                 |       Expert 2  <--- selected
                   token the same]                 +--->  Expert 3  <--- selected
                                                   |       Expert 4
                                                   |       Expert 5
                                                   |       Expert 6
                                                   |       Expert 7
                                                   |       Expert 8
                                                   |
                                              Router picks top-k
                                              (k=2 in micro-Omni)
```

---

## Architecture: Router + Experts

### The Router

The router is just a small linear layer that scores each expert for each token:

```
router_logits = W_router @ x     # (D,) -> (num_experts,)
router_probs = softmax(router_logits)
```

For micro-Omni: `W_router` is a `(D, 8)` matrix. For each token, it produces 8 scores (one per expert), then softmax turns these into probabilities.

Think of the router as a hospital receptionist. The patient (token) describes their symptoms (embedding vector). The receptionist quickly assesses which departments are most relevant and sends the patient to the top 2.

### Top-k Expert Selection

From the 8 probability scores, select the top k (micro-Omni uses k=2):

```
Input token embedding x: (D,)
Router scores:   [0.05, 0.35, 0.02, 0.08, 0.30, 0.10, 0.05, 0.05]
                   E1    E2    E3    E4    E5    E6    E7    E8

Top-2 selection: E2 (0.35) and E5 (0.30)
Renormalize:     E2: 0.35/(0.35+0.30) = 0.538
                 E5: 0.30/(0.35+0.30) = 0.462
```

### Weighted Combination

The selected experts each process the token independently, then their outputs are combined using the renormalized weights:

```
output = 0.538 * Expert2(x) + 0.462 * Expert5(x)
```

Each expert is a standard MLP (or SwiGLU MLP in micro-Omni). They have the same architecture but different learned weights, so they specialize in different patterns.

### Full MoE Forward Pass

```
Token x: (B, T, D)
    |
    v
+-------------------+
| Router            |  W_router: (D, num_experts)
| logits = x @ W   |  -> (B, T, 8) raw scores
| probs = softmax() |  -> (B, T, 8) probabilities
+-------------------+
    |
    v
+-------------------+
| Top-k Selection   |  Select k=2 highest probs per token
| renormalize       |  weights sum to 1.0
+-------------------+
    |
    v
+-------+-------+-------+-------+-------+-------+-------+-------+
| Exp 1 | Exp 2 | Exp 3 | Exp 4 | Exp 5 | Exp 6 | Exp 7 | Exp 8 |
| (MLP) | (MLP) | (MLP) | (MLP) | (MLP) | (MLP) | (MLP) | (MLP) |
+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
    |       |                       |
    |  (only selected experts       |
    |   actually compute)           |
    v                               v
+-----------------------------------+
| Weighted sum of expert outputs    |
| output = w2*E2(x) + w5*E5(x)     |
+-----------------------------------+
    |
    v
Output: (B, T, D)
```

---

## Capacity vs Compute: The MoE Bargain

Here is the magic. With 8 experts and top-2 selection:

| Metric | Dense Model | MoE (8 experts, top-2) |
|--------|-------------|------------------------|
| Total parameters | 1x | ~8x (in FFN layers) |
| Parameters active per token | 1x | ~2x |
| FLOPs per token | 1x | ~2x |
| Model quality | Baseline | Significantly better |

You get 8x the parameters (capacity to learn diverse patterns) but only pay 2x the compute per token. The model is effectively 8x "smarter" for only 2x the cost. The remaining 6 experts sit idle for any given token but are ready for tokens that need them.

Think of it as a law firm with 8 lawyers. Any case only needs 2 lawyers at a time, but the firm can handle any type of case because it has specialists in every area. The firm's total expertise is 8x one lawyer, but the billing per case is only 2x.

---

## The Load Balancing Problem

Without intervention, the router often learns to send everything to one or two "favorite" experts. This is called expert collapse, and it defeats the purpose of MoE.

Why does this happen? Early in training, one expert might be slightly better due to random initialization. The router sends more tokens to it. With more training data, it gets even better. Fewer tokens go to other experts, so they stagnate. Positive feedback loop.

### The Hospital Analogy

Imagine if the receptionist always sent every patient to the same doctor because that doctor happened to handle the first few cases well. The other doctors would never gain experience, and the one overworked doctor would become a bottleneck.

### Auxiliary Loss: Enforcing Balance

The solution is an auxiliary loss that penalizes uneven expert usage. The idea: if expert usage fractions are `[f1, f2, ..., f8]` (where `sum = 1`), then perfect balance gives `fi = 1/8 = 0.125` for all i.

The auxiliary loss encourages this balance. It is added to the main training loss with a small weight (typically 0.01-0.1) so it guides the router without dominating the learning signal.

In practice, you do not need perfect balance. Having some experts used 2x more than others is fine. The goal is preventing collapse where 1-2 experts handle 90%+ of tokens.

---

## Sorted Batched Dispatch: micro-Omni's Optimization

The naive implementation of MoE loops over tokens and routes each one individually. This is slow on GPUs because GPUs excel at large batch operations, not loops.

micro-Omni uses **sorted batched dispatch**: sort all tokens by their assigned expert, then process each expert's batch in one shot.

### Step by Step

```
1. FLATTEN: Merge batch and time dims
   (B, T, D) -> (B*T, D) = (N, D)

2. ROUTE: Get top-k experts and weights for each token
   topk_indices: (N, k)    e.g., token 0 -> [E2, E5]
   topk_probs:   (N, k)         token 1 -> [E1, E3]
                                 token 2 -> [E2, E7]
                                 ...

3. FLATTEN ASSIGNMENTS: Create one list of (token_id, expert_id, weight)
   [(0, E2, 0.54), (0, E5, 0.46), (1, E1, 0.60), (1, E3, 0.40), ...]

4. SORT BY EXPERT ID:
   [(1, E1, 0.60), (0, E2, 0.54), (2, E2, 0.71), (1, E3, 0.40), ...]
    |--- E1 batch --|--- E2 batch -------------|--- E3 batch ---|

5. COUNT tokens per expert:
   E1: 1 token, E2: 2 tokens, E3: 1 token, ...

6. BATCH PROCESS each expert:
   Expert 1: process 1 token  -> output1
   Expert 2: process 2 tokens -> output2a, output2b  (batched!)
   Expert 3: process 1 token  -> output3
   ...

7. SCATTER results back, weighted:
   output[token_0] += 0.54 * E2(token_0) + 0.46 * E5(token_0)
   output[token_1] += 0.60 * E1(token_1) + 0.40 * E3(token_1)
   ...
```

The key insight at step 6: instead of calling Expert 2 twice (once per token), you batch both tokens together and process them in a single forward pass. GPUs love large batches.

### Why Sort?

Sorting by expert ID groups tokens destined for the same expert together in memory. This means:
- One call to each expert with a contiguous batch of tokens
- No scatter/gather overhead between expert calls
- GPU utilization stays high because each expert processes a reasonable batch

Without sorting, you would need complex indexing operations to gather each expert's tokens, which is slower than a single sort.

---

## When to Use MoE

MoE is not always the right choice:

| Scenario | MoE? | Why |
|----------|------|-----|
| Large model, diverse data | Yes | Experts can specialize in different domains |
| Small model, limited data | No | Not enough data to train 8 experts well |
| Latency-critical inference | Maybe | 2x compute per token vs dense, but better quality-per-FLOP |
| Memory-constrained | No | All expert weights must fit in memory, even if only 2 are active |

micro-Omni makes MoE optional: `use_moe=False` by default. When enabled, it uses 8 experts with top-2 selection. This is configured per model:

```python
# In model config:
"use_moe": true,           # Enable MoE
"num_experts": 8,           # Total number of expert FFNs
"num_experts_per_tok": 2    # How many experts each token uses
```

When MoE is disabled, each transformer block uses a single standard MLP (or SwiGLU MLP). When enabled, the MLP is replaced by the MoE layer with N expert MLPs and a router.

---

## Summary

| Concept | Key Point |
|---------|-----------|
| Scaling dilemma | Bigger models are better but slower and more expensive |
| MoE insight | Not every token needs every parameter |
| Router | Small linear layer that scores experts per token |
| Top-k selection | Activate only k experts (k=2), weighted combination |
| Capacity vs compute | 8 experts = 8x params, 2x compute per token |
| Load balancing | Auxiliary loss prevents expert collapse |
| Sorted dispatch | Sort tokens by expert for efficient GPU batching |
| micro-Omni config | Optional (`use_moe=false`), 8 experts, top-2 when enabled |

---

[← Back to Index](00-INDEX.md) | [Previous: Efficient Attention](09-efficient-attention.md) | [Next: Vector Quantization →](11-vector-quantization.md)
