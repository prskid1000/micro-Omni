# Chapter 03: How Neural Networks Learn

[← Previous: Neural Networks Basics](02-neural-networks-basics.md) | [Back to Index](00-INDEX.md) | [Next: Introduction to Transformers →](04-transformers-intro.md)

---

## 🎯 Learning Objectives

By the end of this chapter, you will understand:
- How neural networks learn from data
- What loss functions measure
- How backpropagation computes gradients
- How optimizers update weights
- The complete training loop

---

## 🎓 The Learning Process - From Scratch

### Prerequisites: What You Need to Know First

Before we learn about *training*, let's make sure we understand what we're training:
- A neural network is a collection of artificial neurons (from Chapter 2)
- Each neuron has **weights** (numbers that we can adjust)
- The network makes predictions by passing data through these neurons
- **Goal**: Adjust the weights until predictions are correct

### The Big Picture: Learning is About Adjustment

**Real-life analogy**: Learning to throw a basketball

1. **First throw**: You miss by 3 feet (too far left)
2. **Adjustment**: Next time, aim a bit to the right
3. **Second throw**: You miss by 1 foot (getting closer!)
4. **Adjustment**: Aim even more right
5. **Keep adjusting** until you make the basket

Neural networks learn the **exact same way**!

### Step-by-Step: How a Neural Network Learns

```
ATTEMPT 1:
─────────────────────────────────────────────────────
1. NETWORK makes a guess (prediction)
   Question: "Is this email spam?"
   Network's answer: 0.3 (30% sure it's spam)

2. HUMAN provides correct answer (label)
   True answer: 1.0 (yes, it is 100% spam)

3. CALCULATE HOW WRONG (error)
   Error = True answer - Network's guess
   Error = 1.0 - 0.3 = 0.7
   
   Translation: "You were 0.7 (or 70%) off! That's pretty bad!"

4. ADJUST THE WEIGHTS
   Think: "If I make w₁ bigger, will the answer get closer?"
   Network adjusts: w₁ = 0.5 → 0.52 (slightly increase)
                   w₂ = 0.3 → 0.28 (slightly decrease)
                   ... (adjust all weights)

ATTEMPT 2:
─────────────────────────────────────────────────────
1. NETWORK tries again (with new weights)
   Network's answer: 0.6 (60% sure it's spam)

2. HUMAN: Still 1.0 (yes, spam)

3. ERROR: 1.0 - 0.6 = 0.4 (better! was 0.7, now 0.4)

4. ADJUST again (but smaller changes now)

ATTEMPT 1000:
─────────────────────────────────────────────────────
Network's answer: 0.98 (98% sure - almost perfect!)
Error: 1.0 - 0.98 = 0.02 (very small!)

SUCCESS! Network has learned! ✓
```

### The Key Insight

- **Weights start random** (network knows nothing)
- **Each mistake teaches** (error shows which direction to adjust)
- **Gradually improve** (thousands of small adjustments)
- **Eventually get good** (low error = learned!)

### Why This Works

Imagine you're blindfolded trying to find a ball:
- Someone tells you "you're getting warmer" or "colder"
- You adjust direction based on feedback
- Eventually, you find the ball!

Neural networks:
- Error is the "warmer/colder" feedback
- Weights are the "direction" to move
- Low error is "found it!"

This is exactly how neural networks learn!

---

## 📊 Loss Functions: Measuring Error

A **loss function** (or cost function) measures how wrong the network's predictions are.

### Goal of Training

```
         Bad Predictions          Good Predictions
Loss:    ██████████ (High)   →   ▓ (Low)
         
The network adjusts weights to minimize loss
```

---

### Common Loss Functions

#### 1. **Mean Squared Error (MSE)** - Regression

Used when predicting continuous values.

```
Formula: MSE = (1/n) Σ(yᵢ - ŷᵢ)²

Example: Predicting house prices
True price:     $300,000
Predicted:      $280,000
Error:          $20,000
Squared error:  $400,000,000
```

```python
import torch
import torch.nn as nn

mse_loss = nn.MSELoss()
predictions = torch.tensor([280.0])
targets = torch.tensor([300.0])
loss = mse_loss(predictions, targets)
print(loss)  # tensor(400.0)
```

---

#### 2. **Cross-Entropy Loss** - Classification

Used for classification tasks (spam/not spam, cat/dog, etc.).

```
Example: Classifying an image as cat or dog

Network output (probabilities):
Cat: 0.8  ←─ Confident
Dog: 0.2

True label: Cat

Cross-Entropy Loss = -log(0.8) = 0.223  (Low loss, good!)

If network predicted:
Cat: 0.1
Dog: 0.9  ← Wrong!

Cross-Entropy Loss = -log(0.1) = 2.30  (High loss, bad!)
```

```python
import torch
import torch.nn as nn

ce_loss = nn.CrossEntropyLoss()
# Network outputs (logits)
predictions = torch.tensor([[2.0, 0.5]])  # Before softmax
targets = torch.tensor([0])  # Class 0 (cat)
loss = ce_loss(predictions, targets)
print(loss)  # tensor(0.223)
```

📌 **μOmni uses Cross-Entropy Loss** for text generation (next-token prediction).

---

#### 3. **CTC Loss** - Sequence Alignment

Used when input and output sequences have different lengths.

```
Example: Speech Recognition

Audio frames:     [a][a][b][b][b][c][c]
                   ↓  ↓  ↓  ↓  ↓  ↓  ↓
Transcription:    "abc"

CTC automatically handles:
- Repeated characters
- Variable-length alignment
- Insertions/deletions
```

📌 **μOmni's Audio Encoder** uses CTC Loss for speech-to-text training.

---

## 🔄 Backpropagation: The Learning Algorithm

**Backpropagation** computes how much each weight contributed to the error.

### The Chain Rule

Think of a chain of consequences:

```
You oversleep → Miss bus → Late to work → Boss angry

If you want to fix "Boss angry":
Work backwards through the chain!

Similarly in neural networks:
Input → Layer 1 → Layer 2 → Output → Loss

To minimize loss:
Compute how each layer's weights affected the final loss
```

---

### Visual Walkthrough

```
FORWARD PASS (Compute output and loss):

x=2 ──→ [w=0.5] ──→ z=1 ──→ [σ] ──→ ŷ=0.73 
                                      ↓
                              True y = 1.0
                                      ↓
                            Loss = 0.073

BACKWARD PASS (Compute gradients):

∂L/∂w = ?  ←──────────────────┘
         How much should we change w to reduce loss?

Using chain rule:
∂L/∂w = ∂L/∂ŷ × ∂ŷ/∂z × ∂z/∂w
      = -0.27 × 0.196 × 2
      = -0.106

Interpretation: Decreasing w by 0.106 will reduce loss!
```

---

### Gradient Descent

Once we have gradients, we update weights:

```
New Weight = Old Weight - Learning Rate × Gradient

Example:
w_old = 0.5
gradient = -0.106
learning_rate = 0.1

w_new = 0.5 - (0.1 × -0.106)
      = 0.5 + 0.0106
      = 0.5106

Network improved slightly!
Repeat thousands of times → Converges to good solution
```

---

## 📉 Gradient Descent Visualization

```
        Loss
         ↑
     10  │    ╱╲
         │   ╱  ╲
      5  │  ╱    ╲    ← Start here (w=0.1)
         │ ╱      ╲
      0  ├╱────────╲──→ Weight
         0    0.5    1.0
              ↑
          Goal: Reach minimum!

Training process:
Step 1: w=0.1, loss=8.0  ───→ gradient=+5.0
Step 2: w=0.4, loss=2.0  ───→ gradient=+1.0  
Step 3: w=0.5, loss=0.1  ───→ gradient≈0 (converged!)
```

---

## 🎯 Optimizers: Smart Weight Updates

### 1. **SGD (Stochastic Gradient Descent)**

Basic optimizer: Just follows the gradient.

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Update step:
optimizer.zero_grad()     # Clear old gradients
loss.backward()           # Compute gradients
optimizer.step()          # Update weights
```

**Pros:** Simple, reliable  
**Cons:** Can be slow, sensitive to learning rate

---

### 2. **Adam (Adaptive Moment Estimation)**

Smarter optimizer: Adapts learning rate for each parameter.

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

**How it's better:**
- Keeps moving average of past gradients (momentum)
- Adapts learning rate per parameter
- Usually converges faster than SGD

📌 **μOmni uses Adam or AdamW** optimizer.

---

### 3. **AdamW**

Adam with better weight decay (regularization).

```python
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=1e-4,
    weight_decay=0.01  # Prevents overfitting
)
```

**Weight decay** = Penalty for large weights (encourages simpler models)

---

## 🔁 The Complete Training Loop

### Training Process

```
┌─────────────────────────────────────────────┐
│             TRAINING LOOP                   │
│                                             │
│  For each epoch:                            │
│    For each batch:                          │
│      1. Forward pass  ──→ Predictions       │
│      2. Compute loss  ──→ Error metric      │
│      3. Backward pass ──→ Gradients         │
│      4. Update weights ──→ Improve model    │
│                                             │
│  Repeat until convergence!                  │
└─────────────────────────────────────────────┘
```

---

### Code Example: Complete Training Loop

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Create model, loss, optimizer
model = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 2. Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for batch_inputs, batch_labels in dataloader:
        
        # FORWARD PASS
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_labels)
        
        # BACKWARD PASS
        optimizer.zero_grad()  # Clear old gradients
        loss.backward()        # Compute new gradients
        optimizer.step()       # Update weights
        
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

---

## 📊 Batches and Epochs

### Why Batches?

```
Dataset: 1000 images

Option 1: Process all at once
❌ Requires too much memory
❌ Slow, infrequent updates

Option 2: Process one at a time
❌ Updates too noisy
❌ Slow (can't parallelize)

Option 3: Process in batches ✅
✅ Batch size = 32
✅ 1000 / 32 = 31 batches per epoch
✅ Balanced: stable & efficient
```

### Terminology

| Term | Definition | Example |
|------|------------|---------|
| **Sample** | One training example | One image |
| **Batch** | Group of samples processed together | 32 images |
| **Iteration** | Processing one batch | 1 forward + backward pass |
| **Epoch** | One pass through entire dataset | 31 iterations (if 1000 samples, batch=32) |

```
Epoch 1:  Batch 1 → Batch 2 → ... → Batch 31
Epoch 2:  Batch 1 → Batch 2 → ... → Batch 31
...
Epoch 10: Batch 1 → Batch 2 → ... → Batch 31
```

---

## 📈 Learning Rate: The Most Important Hyperparameter

### What is Learning Rate?

```
Learning Rate = Step size when updating weights

Too Small:              Just Right:           Too Large:
┌────────┐             ┌────────┐            ┌────────┐
│        │             │   ╲    │            │        │
│    ╲   │             │    ╲   │            │  ╲  ╱  │
│     ╲  │             │     ╲  │            │   ╲╱   │
│      ╲ │             │      ╲ │            │   ╱╲   │
│       ╲│             │       ●│            │  ╱  ╲  │
└────────┘             └────────┘            └────────┘
Slow learning         Converges!          Overshoots!
(many epochs)         (optimal)           (diverges)
```

### Learning Rate Scheduling

Start with larger LR, gradually decrease:

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = CosineAnnealingLR(optimizer, T_max=100)

for epoch in range(100):
    train(...)
    scheduler.step()  # Decay learning rate
```

```
Learning Rate Schedule:

LR
 ↑
0.001│╲
     │ ╲___
0.0005│     ╲____
     │          ╲___
0.0001│              ╲____
     └─────────────────────→ Epoch
     0   25   50   75   100

Benefit: Fast progress early, fine-tuning later
```

📌 **μOmni uses:** Warmup (gradual increase) + Cosine decay

---

## ⚠️ Common Training Problems

### 1. **Overfitting**

Model memorizes training data but fails on new data.

```
Training Loss:  ╲_____ (keeps decreasing)
Validation Loss:╲_____╱‾‾‾ (starts increasing!)
                      ↑
                  Overfitting begins
```

**Solutions:**
- Use dropout
- Add weight decay
- Get more training data
- Use data augmentation
- Early stopping

---

### 2. **Underfitting**

Model too simple to learn patterns.

```
Training Loss:   ╲___‾‾‾‾‾ (plateaus high)
Validation Loss: ╲___‾‾‾‾‾ (also high)

Model isn't learning enough!
```

**Solutions:**
- Increase model capacity (more layers/neurons)
- Train longer
- Reduce regularization

---

### 3. **Vanishing Gradients**

Gradients become too small in deep networks.

```
Layer 10:  gradient = 0.5
Layer 9:   gradient = 0.25
Layer 8:   gradient = 0.125
Layer 7:   gradient = 0.0625
...
Layer 1:   gradient = 0.0001  ← Too small to learn!
```

**Solutions:**
- Use ReLU/GELU (not sigmoid)
- Use residual connections (skip connections)
- Use proper initialization
- Use normalization layers (RMSNorm)

📌 **μOmni addresses this with:** RMSNorm, GELU, residual connections

---

### 4. **Exploding Gradients**

Gradients become too large.

```
Layer 1:  gradient = 0.5
Layer 2:  gradient = 2.0
Layer 3:  gradient = 8.0
Layer 4:  gradient = 32.0
Layer 5:  gradient = 128.0  ← NaN! (overflow)
```

**Solutions:**
- Gradient clipping
- Lower learning rate
- Better initialization

```python
# Gradient clipping in PyTorch
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

📌 **μOmni uses:** Gradient clipping (max_norm=1.0)

---

## 🎯 Regularization Techniques

### 1. **Dropout**

Randomly "drop" neurons during training.

```
Training (dropout=0.5):
Input → ● ○ ● ○ ● ○ ● ●  (50% randomly disabled)
         ↓           ↓
       Forces network to learn redundant representations

Inference (dropout off):
Input → ● ● ● ● ● ● ● ●  (All neurons active)
```

```python
class ModelWithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 50)
        self.dropout = nn.Dropout(0.5)  # 50% dropout
        self.fc2 = nn.Linear(50, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)  # Applied only during training
        x = self.fc2(x)
        return x
```

---

### 2. **Weight Decay**

Penalize large weights.

```
Loss_total = Loss_data + λ × Σ(w²)
                         ↑
                    Weight penalty
```

Encourages simpler models (smaller weights = less overfitting).

---

### 3. **Early Stopping**

Stop training when validation loss stops improving.

```
Epoch 1:  val_loss = 1.5  ✓ Best so far
Epoch 2:  val_loss = 1.2  ✓ Best so far
Epoch 3:  val_loss = 1.0  ✓ Best so far
Epoch 4:  val_loss = 1.1  ⚠ Getting worse
Epoch 5:  val_loss = 1.3  ⚠ Still worse
→ STOP! Use model from Epoch 3
```

---

## 💻 Practical Training Tips

### 1. **Monitor Your Training**

```python
# Log important metrics
print(f"Epoch {epoch}:")
print(f"  Train Loss: {train_loss:.4f}")
print(f"  Val Loss: {val_loss:.4f}")
print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
```

### 2. **Save Checkpoints**

```python
# Save best model
if val_loss < best_loss:
    torch.save(model.state_dict(), 'best_model.pt')
    best_loss = val_loss
    print("✓ Saved new best model!")
```

### 3. **Use Validation Set**

```
Total Data: 10,000 samples

Training:   8,000 (80%)  → Train model
Validation: 1,000 (10%)  → Monitor overfitting
Test:       1,000 (10%)  → Final evaluation
```

---

## 💡 Key Takeaways

✅ **Loss function** measures prediction error  
✅ **Backpropagation** computes gradients via chain rule  
✅ **Optimizer** updates weights to minimize loss  
✅ **Training loop**: Forward → Loss → Backward → Update  
✅ **Batch processing** balances efficiency and stability  
✅ **Learning rate** is crucial (use scheduling!)  
✅ **Regularization** prevents overfitting (dropout, weight decay)

---

## 🎓 Self-Check Questions

1. What's the purpose of a loss function?
2. What does backpropagation compute?
3. Why do we use batches instead of processing one sample at a time?
4. What happens if the learning rate is too large?
5. Name three ways to prevent overfitting.

<details>
<summary>📝 Click to see answers</summary>

1. To measure how wrong the network's predictions are (quantify error)
2. Backpropagation computes gradients (how much each weight contributed to the loss)
3. Batches balance memory efficiency, training stability, and parallelization
4. Training becomes unstable, loss may increase, model may diverge (fail to learn)
5. Any three of: dropout, weight decay, early stopping, data augmentation, more training data, simpler model
</details>

---

## ➡️ Next Steps

Now you understand how neural networks learn! Let's explore the revolutionary Transformer architecture.

[Continue to Chapter 04: Introduction to Transformers →](04-transformers-intro.md)

Or return to the [Index](00-INDEX.md) to choose a different chapter.

---

**Chapter Progress:** Foundation ●●●○○ (3/5 complete)

