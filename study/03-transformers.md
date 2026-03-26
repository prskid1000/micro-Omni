[← Previous: 02-neural-networks](02-neural-networks.md) | [Index](00-INDEX.md) | [Next: 04-tokens-embeddings-position →](04-tokens-embeddings-position.md)

# Chapter 03: The Transformer

### Why RNNs Failed

Before Transformers, the dominant architecture for sequential data (text, audio) was
the **Recurrent Neural Network (RNN)**. An RNN processes tokens one at a time, passing
a hidden state from each step to the next -- like a game of telephone.

The problems were severe:

**1. Sequential bottleneck.** Because each step depends on the previous one, you cannot
parallelize. Processing a 1000-word sentence means 1000 sequential steps. GPUs, which
are built for parallel work, sit mostly idle.

**2. Vanishing gradients.** Information from the beginning of a long sequence gets
diluted as it passes through hundreds of steps -- like a whispered message in a game
of telephone that is garbled beyond recognition by the time it reaches the last player.
By the time the gradient flows back to early words, it has shrunk to nearly zero. The
network cannot learn long-range dependencies.

```
  RNN: sequential, one word at a time

  "The"  -->  "cat"  -->  "sat"  -->  "on"  -->  "the"  -->  "mat"
    h0   -->   h1    -->   h2    -->   h3   -->   h4    -->   h5

  Problem: by h5, the information about "The" is faded.
  Problem: step 5 must wait for steps 1-4 to finish.
```

Researchers tried patches (LSTMs, GRUs) that added gates to control information flow.
They helped, but the fundamental sequential bottleneck remained.

---

### The 2017 Breakthrough: "Attention Is All You Need"

In 2017, Vaswani et al. published a paper with a bold claim: throw away recurrence
entirely. Replace it with a mechanism called **attention** that lets every word look at
every other word simultaneously.

The result was the **Transformer** -- an architecture that is:
- **Parallel:** all positions are processed at once, making GPUs happy.
- **Long-range:** any word can directly attend to any other word, no matter how far
  apart. No telephone game.
- **Scalable:** performance improves predictably as you add data and parameters.

Every major AI model since 2018 -- language models, speech recognition, image
generation, and micro-Omni -- is built on Transformers.

---

### Self-Attention: Step by Step

Self-attention is the core mechanism. Here is how it works, explained through a
**library search** analogy.

Imagine you walk into a library with a question. You:
1. Formulate your **Query** (Q) -- "I need information about black holes."
2. Read the **Key** (K) on each book spine -- the title that summarizes what is inside.
3. When a Key matches your Query well, you pull the **Value** (V) -- the actual content
   of that book.

In a Transformer, **every token** simultaneously acts as the searcher (Q) and as a book
on the shelf (K and V). Each token asks: "Which other tokens are relevant to me?"

#### Step 1: Create Q, K, V

Each token's embedding is multiplied by three learned weight matrices to produce three
vectors:

```
  Token embedding (d=384)
       |
       +---> W_Q ---> Query  (what am I looking for?)
       |
       +---> W_K ---> Key    (what do I contain?)
       |
       +---> W_V ---> Value  (what information do I carry?)
```

#### Step 2: Compute Similarity Scores (Dot Product)

Each Query is compared against every Key using a dot product. A high dot product means
"these two tokens are relevant to each other."

```
  Sentence: "The cat sat"

          Key_The   Key_cat   Key_sat
           |          |          |
  Q_The  [ 0.1       0.8        0.3  ]   <-- "The" finds "cat" most relevant
  Q_cat  [ 0.2       0.1        0.9  ]   <-- "cat" finds "sat" most relevant
  Q_sat  [ 0.05      0.7        0.2  ]   <-- "sat" finds "cat" most relevant
```

#### Step 3: Scale by sqrt(d_k)

Before normalizing, we divide each score by the square root of the key dimension
(sqrt(d_k)). Why? Without scaling, when d_k is large, dot products become very large
numbers. Large numbers pushed into softmax produce extremely peaked distributions --
almost all weight on one token, nearly zero on the rest. This makes gradients tiny and
training unstable.

Dividing by sqrt(d_k) keeps the scores in a moderate range where softmax behaves well.

```
  scaled_score = dot_product(Q, K) / sqrt(d_k)

  If d_k = 64:  sqrt(64) = 8
  A raw score of 24.0 becomes 24.0 / 8 = 3.0  (much more manageable)
```

#### Step 4: Softmax -- Normalize to Probabilities

Apply softmax across each row so the attention weights sum to 1. Now each token has a
probability distribution over all other tokens.

```
  Raw scores:    [ 0.1,   0.8,   0.3 ]
                           |
                       (scale by sqrt(d_k))
                           |
  After softmax: [ 0.15,  0.55,  0.30 ]   <-- sums to 1.0
```

#### Step 5: Weighted Sum -- The Answer

Multiply each Value by its attention weight and sum. The result is a new representation
for each token that blends information from all the tokens it found relevant.

```
  output_The = 0.15 * V_The  +  0.55 * V_cat  +  0.30 * V_sat
               ^                 ^                  ^
               (a little of     (mostly cat's      (some of sat's
                itself)          information)        information)
```

#### The Full Equation

```
  Attention(Q, K, V) = softmax( Q * K^T / sqrt(d_k) ) * V
```

One line. That is the entire self-attention mechanism.

---

### Multi-Head Attention: Parallel Search Strategies

A single attention head can only capture one type of relationship. But language is
rich -- you need to track grammar, meaning, coreference, tone, and more simultaneously.

**Multi-head attention** runs several attention heads in parallel, each with its own
Q, K, V weight matrices. Each head learns to focus on a different aspect of the input.

```
  Input (d=384)
    |
    +------+------+------+------+------+------+
    |      |      |      |      |      |      |
  Head1  Head2  Head3  Head4  Head5  Head6
  (d=64) (d=64) (d=64) (d=64) (d=64) (d=64)
    |      |      |      |      |      |
    +------+------+------+------+------+
    |
  Concatenate (6 * 64 = 384)
    |
  Linear projection (d=384 -> d=384)
    |
  Output (d=384)
```

Think of it as sending six research assistants to the library at the same time. One
looks for grammatical relationships, another for semantic similarity, a third for
positional patterns, and so on. Their findings are combined into a single comprehensive
report.

micro-Omni uses **4 heads** with a model dimension of 128, so each head works in a
32-dimensional subspace (128 / 4 = 32).

---

### The Transformer Block

A single Transformer block combines attention with a feed-forward network, wrapped in
normalization and residual connections. Here is the complete data flow:

```
          Input (d=384)
            |
            v
    +------------------+
    |    LayerNorm     |
    +------------------+
            |
            v
    +------------------+
    |  Multi-Head      |
    |  Self-Attention  |
    |  (6 heads)       |
    +------------------+
            |
            +--------> (+) <--- Residual (add original input back)
                        |
                        v
                +------------------+
                |    LayerNorm     |
                +------------------+
                        |
                        v
                +------------------+
                |  Feed-Forward    |
                |  Network (FFN)  |
                |  d -> 4d -> d   |
                |  384->1536->384 |
                |  (with GELU)    |
                +------------------+
                        |
                        +---> (+) <--- Residual (add pre-FFN input back)
                               |
                               v
                        Block Output (d=384)
```

Let's break down the components we have not covered yet:

#### LayerNorm -- Standardize Before Each Sub-layer

LayerNorm rescales the values in each token's vector to have zero mean and unit
variance. Like calibrating a thermometer before each measurement -- it keeps the
numbers in a stable range so training does not diverge.

#### Feed-Forward Network (FFN) -- Think Independently

After attention (which mixes information between tokens), the FFN processes each token
independently. It expands the dimension to 4x (384 to 1536), applies GELU activation,
then compresses back down (1536 to 384).

Think of it as: attention is the group discussion, FFN is the individual reflection
where each token processes what it heard.

---

### Residual Connections: Express Elevators

Notice the "(+)" symbols in the diagram above. These are **residual connections** (also
called skip connections). They add the input of a sub-layer directly to its output.

Why? In a deep network, gradients must flow backwards through every layer during
training. Without residual connections, gradients can vanish or explode as they traverse
many layers -- the same problem that plagued RNNs.

Residual connections provide a **highway** for gradients. Even if a layer's learned
function contributes nearly nothing, the gradient can still flow through the skip
connection untouched.

```
  Without residuals:                 With residuals:

  x -> [Layer] -> [Layer] -> out     x -> [Layer] ---(+)--> [Layer] ---(+)--> out
                                          |          ^       |          ^
  Gradient must pass through              +----------+       +----------+
  every layer (may vanish)                (shortcut: gradient flows directly)
```

The analogy: in a tall building, residual connections are like express elevators. Even
if the stairs (regular layers) are congested, the elevator (skip connection) gets you
straight to the floor you need.

---

### Stacking Blocks: Deeper = More Abstract

A single Transformer block captures local patterns and simple relationships. By
stacking multiple blocks, the network builds progressively more abstract understanding:

```
  Block 1:  "saw" is a verb, "the" is a determiner
  Block 2:  "the cat" is a noun phrase
  Block 3:  "the cat sat on" describes a spatial relationship
  Block 4:  the sentence is in past tense, casual register
    ...
  Block 8:  full semantic understanding of the passage
```

Each layer refines the representation. Early layers handle surface-level features
(syntax, word identity). Later layers capture deep meaning (intent, context, reasoning).

Like reading a sentence multiple times: first you parse the words, then the grammar,
then the meaning, then the implications.

---

### micro-Omni's Thinker

The core reasoning engine of micro-Omni is called the **Thinker**. It is a stack of
Transformer blocks with these specifications (synthetic config):

```
  +------------------------------------------+
  |           micro-Omni Thinker             |
  +------------------------------------------+
  |  Blocks:          4                      |
  |  Attention heads:  4 per block           |
  |  Model dimension: 128 (d_model)          |
  |  Head dimension:  32  (128 / 4)          |
  |  FFN inner dim:   344 (8/3 * 128)        |
  |  Activation:      SwiGLU                 |
  |  Norm:            RMSNorm (pre-norm)     |
  +------------------------------------------+

  Data flow through the complete Thinker:

  Input embeddings (d=128)
       |
       v
  +-----------+
  |  Block 1  | ---+
  +-----------+    | residual
       |<----------+
       v
  +-----------+
  |  Block 2  | ---+
  +-----------+    |
       |<----------+
       v
      ...
       v
  +-----------+
  |  Block 4  | ---+
  +-----------+    |
       |<----------+
       v
  Final RMSNorm
       |
       v
  Output (d=128)
       |
       +---> Text head (vocabulary logits)
       +---> Audio head (audio features)
```

Four blocks, four heads each, dimension 128. Small enough to train on one GPU, deep
enough to understand the interplay of text, images, and audio. In the coming chapters,
we will see how the different modalities are encoded and fed into this Thinker, and
what advanced techniques (like Grouped Query Attention and Mixture of Experts) can
make it even more efficient.
