[← Previous: 01-what-is-ai](01-what-is-ai.md) | [Index](00-INDEX.md) | [Next: 03-transformers →](03-transformers.md)

# Chapter 02: Neural Networks & How They Learn

### What a Neuron Does

A single artificial neuron is a tiny decision-maker. Think of it as a **vote-counting
machine** at an election:

1. **Inputs** arrive (like votes from different districts).
2. Each input is multiplied by a **weight** (how much that district matters).
3. All weighted votes are summed, and a **bias** is added (a built-in preference).
4. The total passes through an **activation function** that decides the final output.

```
  x1 --(*w1)--\
                \
  x2 --(*w2)----+---> [ sum + bias ] ---> [ activation ] ---> output
                /
  x3 --(*w3)--/

  Mathematically:  output = activation( x1*w1 + x2*w2 + x3*w3 + bias )
```

The **weights** and **bias** are the knobs the network learns to adjust. At the start
they are random. By the end of training, they encode everything the model "knows."

---

### Activation Functions: The Gate Keepers

Without an activation function, stacking layers would just be multiplying matrices --
no matter how many layers, the result is still a single linear transformation. Activation
functions add the non-linearity that lets networks learn complex patterns.

#### ReLU (Rectified Linear Unit) -- The Simple Gate

Rule: if the input is positive, pass it through; if negative, output zero.

```
  output
    |          /
    |         /
    |        /
    |       /
  --+------/---------> input
    |  0  0
    |
  ReLU: max(0, x)
```

Like a one-way valve in plumbing -- water flows forward, but nothing flows back. Fast,
simple, works well in most networks.

#### GELU (Gaussian Error Linear Unit) -- The Smooth Gate

Instead of a hard cutoff at zero, GELU lets a small amount of negative values through,
with the probability based on how negative they are.

```
  output
    |           /
    |          /
    |        _/
    |      _/
  --+----~/----------> input
    |  ~0
    |_/
  GELU: x * P(X <= x)   (smooth curve near zero)
```

Like a dimmer switch instead of an on/off switch -- smoother gradients during training.
micro-Omni uses SwiGLU (which internally uses the Swish activation) in its feed-forward
layers because it trains more stably and produces better results.

---

### Layers: The Assembly Line

A neural network is neurons organized into layers, like stages on a factory assembly
line:

```
  Input Layer       Hidden Layers        Output Layer
  (raw data)     (feature extraction)    (prediction)

  [ o ]           [ o ]   [ o ]           [ o ]
  [ o ] -------> [ o ] -> [ o ] -------> [ o ]
  [ o ]           [ o ]   [ o ]           [ o ]
  [ o ]           [ o ]   [ o ]

  Stage 1:        Stage 2:  Stage 3:      Stage 4:
  Receive         Find       Combine       Make
  ingredients     edges,     features      final
                  shapes     into          dish
                             concepts
```

- **Input layer:** receives raw numbers (pixel values, audio samples, word codes).
- **Hidden layers:** each extracts progressively more abstract features. Early layers
  find edges; later layers find faces.
- **Output layer:** produces the final answer (a word probability, a classification).

More hidden layers = "deeper" network = "deep" learning.

---

### The Forward Pass

The forward pass is simply data flowing left-to-right through the network, layer by
layer, until a prediction comes out the other end. No learning happens here -- just
computation.

```
  Input: "The cat sat on the ___"

  [Embedding] -> [Layer 1] -> [Layer 2] -> ... -> [Layer 8] -> [Output Head]
                                                                     |
                                                                     v
                                                              Prediction: "mat"
```

Think of it as pushing raw ingredients through the assembly line to see what dish
comes out. If the dish is wrong, we need to figure out what went wrong -- that is
where the loss function comes in.

---

### Loss Functions: How Wrong Is the Prediction?

A loss function is a **scoreboard** that measures the gap between what the model
predicted and what the correct answer was. Lower loss = better model.

| Loss Function | Used For | Analogy |
|---------------|----------|---------|
| **Cross-Entropy** | Text prediction (next token) | "How surprised are you by this answer?" High surprise = high loss |
| **CTC (Connectionist Temporal Classification)** | Speech-to-text alignment | "Match this audio to this transcript, even when timing is fuzzy" -- like matching subtitles to a movie without exact timestamps |
| **MSE (Mean Squared Error)** | Reconstruction (audio waveforms) | "How far off is each predicted point from the real point?" Average of all the squared errors |

micro-Omni uses all three: cross-entropy for its text predictions, CTC for aligning
audio input to text, and MSE for reconstructing audio output.

---

### Backpropagation: Tracing Blame Backwards

Imagine a factory produces a defective product. The quality inspector at the end (loss
function) catches it. Now the factory manager walks **backwards** through the assembly
line asking: "Who contributed to this defect, and by how much?"

That is backpropagation. It computes the **gradient** -- how much each weight
contributed to the error -- by applying the chain rule of calculus from the output
layer back to the input layer.

```
  Forward:   Input ---> Layer 1 ---> Layer 2 ---> Output ---> Loss
                                                                |
  Backward:  Input <--- Layer 1 <--- Layer 2 <--- Output <--- Loss
             (adjust    (adjust      (adjust      (how wrong?)
              weights)   weights)     weights)
```

Each weight gets a gradient: a number that says "if you increase this weight slightly,
the loss changes by this much." The sign tells the direction, the magnitude tells how
much it matters.

---

### Gradient Descent: Rolling Downhill

Once we have gradients, we update the weights to **reduce** the loss. Picture a ball
on a hilly landscape. The ball rolls downhill toward the valley (minimum loss).

```
  Loss
   ^
   |  \
   |   \       /
   |    \     /
   |     \   /
   |      \_/   <-- we want to reach this valley
   +-------------------> weight value
        ^
        ball rolls this way (opposite to gradient)
```

The update rule is simple:

```
  new_weight = old_weight - learning_rate * gradient
```

We subtract because we want to move **opposite** to the direction that increases loss.

---

### Optimizers: Smarter Ways to Roll Downhill

Plain gradient descent treats every weight the same. Modern optimizers are smarter:

| Optimizer | Idea | Analogy |
|-----------|------|---------|
| **SGD** | Update using a small random batch of data | Walking downhill with a rough compass -- noisy but works |
| **Adam** | Track both the average gradient and its variance, adapt per-weight | A hiker with GPS and speed tracking -- adjusts step size for each weight |
| **AdamW** | Adam + proper weight decay (penalizes large weights) | GPS hiker who also carries a lighter pack over time -- prevents overgrowth |

micro-Omni uses **AdamW** because it combines adaptive learning rates with clean
regularization, which is the standard choice for Transformer training.

---

### Learning Rate: The Goldilocks Problem

The learning rate controls how big each update step is.

```
  Too HIGH (lr = 0.1)          Just RIGHT (lr = 0.001)       Too LOW (lr = 0.000001)

  Loss                         Loss                          Loss
   |  /\  /\                    |  \                           | \
   | /  \/  \  (bouncing!)      |   \                          |  \
   |/        \/                 |    \_____  (smooth!)         |   \_______ (barely
   +------------>               +------------>                 +------------>  moving)
       steps                        steps                          steps
```

- **Too high:** the model overshoots the valley, bouncing around or diverging.
- **Too low:** the model inches forward so slowly it might never reach a good solution
  in practical time.
- **Just right:** smooth convergence to a low loss.

---

### Warmup + Cosine Decay Schedule

In practice, we don't keep the learning rate fixed. micro-Omni uses a schedule with
two phases:

**Phase 1 -- Warmup:** Start with a very small learning rate and ramp up linearly over
the first few hundred steps. This prevents the randomly-initialized model from making
wild updates at the very start.

**Phase 2 -- Cosine Decay:** After warmup, gradually decrease the learning rate
following a cosine curve, gently landing near zero by the end of training.

```
  Learning
  Rate
    ^
    |        ____
    |       /    \
    |      /      \
    |     /        \___
    |    /              \____
    |   /                    \___
    |  /                         \__
    | /                             \
    +--+------------------------------> steps
      |   |
      0  warmup                     end
          ends

      Phase 1:        Phase 2:
      Linear          Cosine
      Warmup          Decay
```

This combination gives the model a safe start and a gentle finish, avoiding both
early instability and late-stage overshoot.

---

### Overfitting and Underfitting

Two failure modes every model can suffer from:

**Underfitting** -- the model is too simple or undertrained. It has not learned enough
patterns. Like a student who only skimmed the textbook and fails the exam.

**Overfitting** -- the model memorized the training data instead of learning general
patterns. It aces the practice test but fails the real exam. Like a student who
memorized every practice answer word-for-word but cannot handle a rephrased question.

```
  Underfitting               Good fit                Overfitting
  (too simple)               (just right)            (memorized noise)

    o   o                    o   o                     o   o
  o   o   o                o   o   o                .o...o...o.
  ----------  (straight    ---/-----\---            ./ \./.\. /\.
              line misses   /         \             (wiggly line
              the pattern)                          hits every point)
```

The goal is the middle ground: learn the real pattern, ignore the noise.

---

### Regularization: Fighting Overfitting

#### Dropout -- Randomly Disable Neurons

During each training step, randomly "turn off" a fraction of neurons (say 10%). This
forces the network to not rely on any single neuron too much -- like a sports team
that practices with random players sitting out so everyone learns to contribute.

```
  Without dropout:          With dropout (training):
  [ o ]--[ o ]--[ o ]      [ o ]--[ X ]--[ o ]     X = disabled
  [ o ]--[ o ]--[ o ]      [ o ]--[ o ]--[ X ]
  [ o ]--[ o ]--[ o ]      [ X ]--[ o ]--[ o ]
```

At test time, all neurons are active but their outputs are scaled down to compensate.

#### Weight Decay -- Penalize Large Weights

Add a small penalty to the loss for having large weight values. This encourages the
model to find simpler solutions. Like a tax on complexity -- the model pays a cost
for every large weight it keeps, so it only keeps the ones that really matter.

micro-Omni uses a weight decay of 0.01 through AdamW.

---

### Key Training Terms

| Term | Definition | Analogy |
|------|-----------|---------|
| **Batch** | A small group of training examples processed together (e.g., 32 sentences) | A tray of cookies going into the oven at once |
| **Epoch** | One complete pass through the entire training dataset | Reading the entire textbook once cover-to-cover |
| **Iteration** (step) | One weight update from one batch | One homework problem |
| **Gradient Accumulation** | Process several batches, sum their gradients, then do one update | Collecting feedback from multiple reviewers before making a revision -- simulates a larger batch when GPU memory is limited |

```
  1 Epoch = all batches processed once

  Dataset: [====|====|====|====|====]
            B1    B2    B3    B4   B5     <-- 5 batches per epoch
            |     |     |     |     |
            v     v     v     v     v
         step1 step2 step3 step4 step5    <-- 5 iterations per epoch

  With gradient accumulation of 2:
            B1    B2    B3    B4   B5
            \    /      \    /     |
           step1       step2    step3     <-- 3 effective updates
```

---

### micro-Omni's Parameter Count

micro-Omni (synthetic config) has approximately **13.9 million** trainable parameters,
distributed across the Thinker (core LLM), Audio Encoder, Vision Encoder, Talker,
and RVQ codec. The Thinker holds the majority of parameters.

For reference, this fits comfortably in the memory of a single consumer GPU. The entire
model's weights file is small enough to email. Yet as you will see in the coming
chapters, careful architecture design makes these parameters surprisingly capable.
