[← Previous: Index](00-INDEX.md) | [Index](00-INDEX.md) | [Next: 02-neural-networks →](02-neural-networks.md)

# Chapter 01: What is AI?

### What AI Actually Means

Artificial Intelligence is not about sentient robots or sci-fi overlords. At its core,
AI is software that takes in data, finds patterns, and makes decisions -- things that
would normally require a human brain. A spam filter reading your email, a thermostat
learning your schedule, a voice assistant understanding "set a timer for 5 minutes" --
all AI.

The key idea: instead of a programmer writing every rule by hand ("if email contains
'free money', mark as spam"), the machine **learns the rules from examples**.

---

### The Hierarchy: AI, Machine Learning, Deep Learning

Think of learning to cook.

| Level | Cooking Analogy | What It Means |
|-------|----------------|---------------|
| **AI** | The entire kitchen | Any technique that makes a machine act "smart" |
| **Machine Learning (ML)** | Following recipes you improve over time | A subset of AI where the machine learns from data |
| **Deep Learning (DL)** | Developing instinct after cooking thousands of meals | A subset of ML using neural networks with many layers |

AI is the broadest umbrella. Machine Learning is the most successful approach inside
that umbrella. Deep Learning is the rocket fuel inside Machine Learning that powers
modern breakthroughs like ChatGPT, image generators, and speech recognition.

```
+-----------------------------------------------+
|                  AI                            |
|   (any system that acts "smart")               |
|                                                |
|   +---------------------------------------+   |
|   |        Machine Learning               |   |
|   |   (learns patterns from data)         |   |
|   |                                       |   |
|   |   +-------------------------------+   |   |
|   |   |      Deep Learning            |   |   |
|   |   |  (many-layered neural nets)   |   |   |
|   |   |                               |   |   |
|   |   |   +---------------------+     |   |   |
|   |   |   | Transformers /      |     |   |   |
|   |   |   | micro-Omni lives    |     |   |   |
|   |   |   | here                |     |   |   |
|   |   |   +---------------------+     |   |   |
|   |   +-------------------------------+   |   |
|   +---------------------------------------+   |
+-----------------------------------------------+
```

---

### Three Ways Machines Learn

Every ML system falls into one (or a blend) of three paradigms.

#### 1. Supervised Learning -- "Teacher Gives Answers"

Imagine a cooking class where the chef shows you a finished dish (the label) and the
recipe (the input) every time. You study hundreds of examples until you can predict the
dish from the recipe on your own.

- **Input:** photo of a cat, **Label:** "cat"
- The model sees thousands of labeled examples and learns to map inputs to outputs.
- Most of micro-Omni's training is supervised: given text, predict the next word; given
  audio, predict the transcript.

#### 2. Unsupervised Learning -- "Find Patterns Alone"

You walk into a grocery store with no signs. Nobody tells you what goes where, but
after a while you notice fruits cluster together, meats cluster together, dairy clusters
together. You discovered the structure yourself.

- No labels, just raw data.
- The model finds hidden groupings, patterns, or compressed representations.
- Example: clustering customer purchase data to find market segments.

#### 3. Reinforcement Learning -- "Trial and Error with Rewards"

A toddler touches a hot stove (negative reward) and learns not to do it again. They
share a toy and get a smile (positive reward) and learn to share more. No one gave a
textbook -- just feedback after each action.

- An agent takes actions in an environment.
- It receives rewards or penalties.
- Over time, it learns a strategy (policy) that maximizes total reward.
- Example: a game-playing AI that learns chess by playing millions of games against
  itself.

```
Supervised        Unsupervised        Reinforcement
----------        ------------        -------------
Input + Label     Input only          Action -> Reward
  |                  |                    |
  v                  v                    v
"This IS a cat"   "These look alike"  "That move won!"
```

---

### Why Deep Learning Won

Three ingredients came together around 2012-2017 that made Deep Learning dominate:

**1. Big Data.** The internet exploded with text, images, and video. Suddenly there
were billions of training examples available for free.

**2. GPUs.** Graphics cards, originally built for video games, turned out to be perfect
for the massive parallel math that neural networks need. Training that would take months
on a CPU takes hours on a GPU.

**3. Transformers.** In 2017, a new architecture called the Transformer replaced older
designs (RNNs, LSTMs). Transformers process all words in a sentence simultaneously
instead of one-by-one, making them faster and better at capturing long-range patterns.
Nearly every modern AI system -- large language models, speech recognition,
image generation -- is built on Transformers.

```
Before 2012          2012-2016             2017-present
-----------          ---------             ------------
Hand-crafted     +   Big Data          +   Transformers
rules, small     |   GPU training      |   Attention mechanism
datasets         |   CNNs for images   |   Scales to billions
                 |   RNNs for text     |   of parameters
                 v                     v
            "Deep Learning era"   "Transformer era"
```

---

### Where micro-Omni Fits

micro-Omni is a **tiny multimodal AI**. Let's unpack those words:

- **Tiny:** only ~13.9 million parameters (synthetic config). For comparison, large
  language models have billions of parameters. micro-Omni is designed to train and
  run on a single consumer GPU.

- **Multimodal:** it handles multiple types of data -- text, images, and audio -- in
  one unified model. Most AIs specialize in just one modality. micro-Omni weaves them
  together.

- **AI (Deep Learning, Transformer-based):** it sits in the innermost ring of our
  hierarchy diagram above. It uses the Transformer architecture, the same family that
  powers the largest models in the world, just scaled down to be accessible.

```
           micro-Omni at a glance
  +--------------------------------------+
  |  Text  ----+                         |
  |             |                        |
  |  Image ----+--->  Transformer  ---> Output
  |             |     ("Thinker")       (text,
  |  Audio ----+      4 blocks           audio,
  |                   ~13.9M params     or both)
  +--------------------------------------+
         Runs on 1 GPU, trains in hours
```

The goal of this study guide is to take you from zero to understanding every piece of
that diagram. Next up: what those "blocks" and "parameters" actually are.
