# Chapter 02: Neural Networks Fundamentals

[← Previous: What is AI?](01-what-is-ai.md) | [Back to Index](00-INDEX.md) | [Next: How Neural Networks Learn →](03-training-basics.md)

---

## 🎯 Learning Objectives

By the end of this chapter, you will understand:
- What an artificial neuron is and how it works
- How neurons connect to form networks
- Different types of layers in neural networks
- The forward pass: how data flows through a network

---

## 🧠 Biological Inspiration

### The Real Neuron

The human brain contains ~86 billion neurons connected by ~100 trillion synapses.

```
Biological Neuron:

    Dendrites          Cell Body         Axon           Synapses
       ↓                   ↓               ↓                ↓
   ────┬────          ┌────────┐       ─────────      ────┬────
       │              │   ○    │              │            │
   ────┼────    →     │        │    →     ────┴────   →  ──┼──
       │              │        │                           │
   ────┴────          └────────┘                       ────┴────

   Inputs         Processes Info    Transmits      Connects to
                                    Signal         other neurons
```

**How it works:**
1. **Dendrites** receive signals from other neurons
2. **Cell body** processes these signals
3. If total signal exceeds threshold → **fires** signal down axon
4. **Synapses** pass signal to connected neurons

---

## ⚡ The Artificial Neuron (Perceptron)

An artificial neuron mimics this behavior mathematically.

### Structure

```
          x₁ ───w₁─┐
                   │
          x₂ ───w₂─┤    ┌─────────┐         ┌──────────┐
                   ├───→│   Sum   │────z───→│Activation│────→ output
          x₃ ───w₃─┤    │ Σ(wᵢxᵢ) │         │ f(z)     │
                   │    └─────────┘         └──────────┘
          b ────┘
         (bias)

Inputs    Weights    Weighted Sum     Activation    Output
```

### Mathematical Formula

For a single neuron:

```
z = w₁x₁ + w₂x₂ + w₃x₃ + ... + b

output = f(z)
```

Where:
- **xᵢ** = inputs (features)
- **wᵢ** = weights (importance of each input)
- **b** = bias (shift/threshold adjustment)
- **f** = activation function

---

## 💻 Concrete Example: Spam Detection Neuron

Let's build a neuron that detects spam emails!

### Inputs (Features)

```python
# Email: "Win free money now!"
x1 = 3  # Number of exclamation marks
x2 = 2  # Number of "money" words (win, money)
x3 = 1  # Contains "free"? (1=yes, 0=no)
```

### Weights (Learned Importance)

```python
w1 = 0.5   # Exclamation marks matter somewhat
w2 = 2.0   # Money words are strong spam signals
w3 = 1.5   # "Free" is a strong indicator
b = -3.0   # Bias (threshold adjustment)
```

### Computation

```python
# Step 1: Weighted sum
z = (w1 * x1) + (w2 * x2) + (w3 * x3) + b
z = (0.5 * 3) + (2.0 * 2) + (1.5 * 1) + (-3.0)
z = 1.5 + 4.0 + 1.5 - 3.0
z = 4.0

# Step 2: Activation (Sigmoid function)
output = 1 / (1 + e^(-z))
output = 1 / (1 + e^(-4.0))
output = 0.98  # ~98% probability of spam!
```

✅ **Email classified as SPAM!**

---

## 🔥 Activation Functions

Activation functions add **non-linearity** to the network, allowing it to learn complex patterns.

### Common Activation Functions

#### 1. **Sigmoid**

```
       1 |         ┌────────
         |       ╱
f(z) =   |     ╱
1/(1+e⁻ᶻ)|   ╱
       0 |─╱─────────────
         └───────────────
        -∞      0      +∞

Output: (0, 1)
Use: Binary classification, gates in LSTM
```

#### 2. **ReLU (Rectified Linear Unit)**

```
       ∞ |        ╱
         |       ╱
f(z) =   |      ╱
max(0,z) |     ╱
       0 |────╱───────
         └─────────────
        -∞   0      +∞

Output: [0, ∞)
Use: Most hidden layers (fast, simple)
```

#### 3. **GELU (Gaussian Error Linear Unit)**

```
       ∞ |         ╱
         |        ╱
         |       ╱
         |     ╱╱
       0 |───╱─────────
         └─────────────
        -∞   0      +∞

Output: (-∞, ∞) but smoother than ReLU
Use: Transformers, modern architectures
```

📌 **μOmni uses**:
- **GELU** in most layers (smooth, effective)
- **SwiGLU** in feedforward layers (advanced variant)

---

## 🏗️ Building a Neural Network

A **neural network** is layers of neurons connected together.

### Simple 3-Layer Network

```
INPUT LAYER    HIDDEN LAYER    OUTPUT LAYER

    x₁ ●────────● h₁ ●─────────● y₁
              ╱  ╲  ╱ ╲
    x₂ ●────●────● h₂ ●───────●
          ╱  ╲  ╱  ╱ ╲        
    x₃ ●───────● h₃ ●─────────● y₂
      
     3 inputs   3 hidden    2 outputs
                neurons
```

**Each connection has a weight!**

---

## 🎯 Types of Layers

### 1. **Dense/Fully Connected Layer**

Every neuron connects to every neuron in the next layer.

```python
# PyTorch example
import torch.nn as nn

layer = nn.Linear(in_features=10, out_features=5)
# Input: 10 neurons → Output: 5 neurons
# Total weights: 10 × 5 = 50 weights + 5 biases = 55 parameters
```

**Used in:** Most neural networks, including μOmni's projectors

---

### 2. **Convolutional Layer**

Slides a filter over input to detect patterns (mainly for images/audio).

```
Input Image:        Filter:         Output (Feature Map):
┌──┬──┬──┬──┐      ┌──┬──┐        ┌──┬──┬──┐
│  │  │  │  │      │ 1│ 0│        │  │  │  │
├──┼──┼──┼──┤      ├──┼──┤        ├──┼──┼──┤
│  │██│██│  │  *   │ 0│ 1│   →    │  │██│  │
├──┼──┼──┼──┤      └──┴──┘        ├──┼──┼──┤
│  │██│██│  │                     │  │██│  │
├──┼──┼──┼──┤                     └──┴──┴──┘
│  │  │  │  │       Detects
└──┴──┴──┴──┘       edges/patterns
```

**Used in:** μOmni's Audio Encoder (ConvDown), image processing

---

### 3. **Embedding Layer**

Converts discrete tokens (words, codes) into continuous vectors.

```
Token ID    →    Dense Vector (Embedding)

   5        →    [0.23, -0.45, 0.67, 0.12, ...]
"cat"       →    [0.1, 0.3, -0.2, 0.5, ...]
  42        →    [-0.3, 0.8, 0.1, -0.6, ...]

Vocabulary size: 5000 words
Embedding dimension: 256

Parameters: 5000 × 256 = 1,280,000 embeddings
```

**Used in:** μOmni's Thinker (token embeddings), RVQ codebooks

---

### 4. **Normalization Layer**

Stabilizes training by normalizing activations.

```
Before:                After (RMSNorm):
[-100, 50, 200, 10] →  [-0.5, 0.3, 1.2, 0.1]

Prevents:
- Exploding gradients (values too large)
- Vanishing gradients (values too small)
```

**Used in:** μOmni uses **RMSNorm** throughout

---

## 📊 Layer Sizes and Parameters

### Understanding Parameter Count

```python
# Example: Dense layer
input_size = 256
output_size = 512

# Parameters:
weights = 256 × 512 = 131,072
biases = 512
total = 131,584 parameters

# In PyTorch:
layer = nn.Linear(256, 512)
print(sum(p.numel() for p in layer.parameters()))
# Output: 131584
```

### μOmni's Parameter Breakdown

| Component | Approximate Parameters |
|-----------|----------------------|
| Thinker (LLM) | ~60-80M |
| Audio Encoder | ~10-15M |
| Vision Encoder | ~15-20M |
| Talker | ~10-15M |
| RVQ Codec | ~100K |
| **Total** | **~120-140M** |

💡 For comparison:
- GPT-3: 175 **billion** parameters
- LLaMA 7B: 7 **billion** parameters
- μOmni: 140 **million** parameters (1000x smaller!)

---

## 🔄 The Forward Pass

### Data Flow Through a Network

```
Step-by-step example:

1. INPUT: x = [1.0, 2.0, 3.0]
           ↓
2. LAYER 1 (Dense): W₁ · x + b₁
   → [0.5, -0.3, 0.8, 1.2]
           ↓
3. ACTIVATION: ReLU
   → [0.5, 0.0, 0.8, 1.2]  (negative → 0)
           ↓
4. LAYER 2 (Dense): W₂ · h₁ + b₂
   → [2.1, 0.7]
           ↓
5. ACTIVATION: Sigmoid
   → [0.89, 0.67]
           ↓
6. OUTPUT: Probabilities for 2 classes
```

### Code Example

```python
import torch
import torch.nn as nn

# Define a simple network
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(3, 4)   # 3 inputs → 4 hidden
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(4, 2)   # 4 hidden → 2 outputs
        self.act2 = nn.Sigmoid()
    
    def forward(self, x):
        x = self.layer1(x)     # Dense layer
        x = self.act1(x)       # Activation
        x = self.layer2(x)     # Dense layer
        x = self.act2(x)       # Activation
        return x

# Use it
model = SimpleNet()
input_data = torch.tensor([1.0, 2.0, 3.0])
output = model(input_data)
print(output)  # → tensor([0.89, 0.67])
```

---

## 🎨 Visualizing What Networks Learn

### Layer-by-Layer Learning

For image classification:

```
INPUT IMAGE: Photo of a cat

LAYER 1 (Shallow): Learns basic features
┌──────────────────────────┐
│  Edges, lines, curves    │
│  / \ | — ○               │
└──────────────────────────┘

LAYER 2 (Middle): Learns combinations
┌──────────────────────────┐
│  Textures, patterns      │
│  Fur, stripes, spots     │
└──────────────────────────┘

LAYER 3 (Deep): Learns parts
┌──────────────────────────┐
│  Eyes, ears, nose        │
│  Body parts              │
└──────────────────────────┘

LAYER 4 (Deeper): Learns objects
┌──────────────────────────┐
│  Complete cat face       │
│  Cat body, cat poses     │
└──────────────────────────┘

OUTPUT: "Cat" (with 95% confidence)
```

---

## 📐 Network Architectures

### Types by Structure

#### 1. **Feedforward Network**

```
Simple one-direction flow:

Input → Hidden → Hidden → Output
  ↓       ↓        ↓        ↓
[Data flows only forward, no loops]
```

#### 2. **Recurrent Network (RNN)**

```
Has loops for sequential data:

     ┌─────┐
     ↓     │
Input → Hidden → Output
        ↑    ↓
        └────┘
[Loops allow memory of previous inputs]
```

#### 3. **Transformer Network** ⭐

```
Uses attention mechanism (parallel processing):

Input tokens
    ↓
Self-Attention (all tokens interact)
    ↓
Feedforward
    ↓
Output

[μOmni's Thinker uses this!]
```

---

## 💪 Network Capacity

### Depth vs Width

```
SHALLOW & WIDE:
┌─●─●─●─●─●─●─●─●─●─●─┐
│                      │
└─●─●─●─●─●─●─●─●─●─●─┘

DEEP & NARROW:
┌─●─●─┐
├─●─●─┤
├─●─●─┤
├─●─●─┤
├─●─●─┤
├─●─●─┤
└─●─●─┘
```

**General Rule:**
- **Deeper** = Can learn more complex, hierarchical patterns
- **Wider** = More capacity within each level
- **Modern trend:** Deep networks (10-100+ layers)

📌 **μOmni's Thinker:** 4 layers, 256 dimensions (tiny by modern standards!)

---

## 🔢 Parameter Calculation Exercise

Calculate parameters for this network:

```
Layer 1: Linear(100, 50)
Layer 2: Linear(50, 25)
Layer 3: Linear(25, 10)
```

<details>
<summary>💡 Click for solution</summary>

```
Layer 1: (100 × 50) + 50 = 5,050
Layer 2: (50 × 25) + 25 = 1,275
Layer 3: (25 × 10) + 10 = 260

Total: 6,585 parameters
```
</details>

---

## 💡 Key Takeaways

✅ **Artificial neuron** = Weighted sum + activation function  
✅ **Neural network** = Layers of neurons connected together  
✅ **Activation functions** add non-linearity (ReLU, GELU, Sigmoid)  
✅ **Types of layers**: Dense, Convolutional, Embedding, Normalization  
✅ **Forward pass** = Data flowing through layers to produce output  
✅ **Deep networks** can learn hierarchical, complex patterns

---

## 🎓 Self-Check Questions

1. What are the three main components of an artificial neuron?
2. Why do we need activation functions?
3. What's the difference between a shallow and deep network?
4. How many parameters does a Linear(10, 5) layer have?
5. What type of layer converts token IDs to vectors?

<details>
<summary>📝 Click to see answers</summary>

1. Inputs (x), Weights (w), Bias (b), and activation function f
2. To add non-linearity, allowing networks to learn complex, non-linear patterns
3. Shallow has few layers; deep has many layers (can learn hierarchical patterns)
4. (10 × 5) + 5 = 55 parameters (50 weights + 5 biases)
5. Embedding layer
</details>

---

## ➡️ Next Steps

Now you know how neural networks are structured. But how do they learn?

[Continue to Chapter 03: How Neural Networks Learn →](03-training-basics.md)

Or return to the [Index](00-INDEX.md) to choose a different chapter.

---

**Chapter Progress:** Foundation ●●○○○ (2/5 complete)

