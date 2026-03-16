# ML Architecture Primer

**For the LLM Paper Reading Group (GPT-1 through InstructGPT)**

> **How to read this document:** Think of building a house. Perceptrons are bricks, MLPs are walls, attention is the architectural plan, transformers are the full house, and RLHF is the interior design that makes it livable. Each section builds on the one before it, so by the end, you'll understand how GPT models work -- from the ground up.

---

## Useful Links

- [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course) — A free, comprehensive introduction to ML fundamentals with interactive visualizations and exercises.
- [Google ML Crash Course — Neural Networks](https://developers.google.com/machine-learning/crash-course/neural-networks) — The neural networks section specifically, covering perceptrons, hidden layers, and backpropagation.
- [TensorFlow Playground](https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.64365&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false) — An interactive tool to experiment with neural networks in the browser. Adjust layers, neurons, activation functions, and learning rate to see how they affect classification in real time.
- [Neural Designer — Neural Network Tutorial](https://www.neuraldesigner.com/learning/tutorials/neural-network/) — A visual tutorial walking through neural network architecture, training, and applications.
- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) — An interactive visualization that lets you see how a Transformer processes text step by step, from input tokens through attention and feed-forward layers to output predictions.

---

## Table of Contents

1. [What is Machine Learning?](#1-what-is-machine-learning)
2. [The Perceptron](#2-the-perceptron)
3. [Multi-Layer Perceptron (MLP)](#3-multi-layer-perceptron-mlp)
4. [Activation Functions](#4-activation-functions)
5. [Loss Functions & Optimization](#5-loss-functions--optimization)
6. [Sequence Models: RNNs](#6-sequence-models-rnns)
7. [LSTMs & GRUs](#7-lstms--grus)
8. [The Attention Mechanism](#8-the-attention-mechanism)
9. [The Transformer](#9-the-transformer)
10. [Decoder-Only Transformers (GPT)](#10-decoder-only-transformers-gpt)
11. [Fine-Tuning & Transfer Learning](#11-fine-tuning--transfer-learning)
12. [RLHF](#12-rlhf)

---

## 1. What is Machine Learning?

Traditional programming is like writing a recipe: you tell the computer exactly what steps to follow. Machine learning flips this around -- you give the computer examples of what you want, and it figures out the recipe on its own.

### Supervised vs. Unsupervised Learning

**Supervised learning** is like a teacher grading flashcards. You show the model an input (the question side) and the correct output (the answer side), and it learns the pattern.

- *Everyday example:* Showing a child photos of dogs and cats, each labeled, until they can identify new animals on their own.

**Unsupervised learning** is like sorting a pile of buttons without instructions. The model finds natural groupings or patterns in data without being told what to look for.

- *Everyday example:* A grocery store grouping customers by shopping habits -- nobody told the algorithm what the groups should be.

### Training vs. Inference

> **Analogy:** Training is like studying for an exam. Inference is like taking the exam.

During **training**, the model sees thousands (or billions) of examples and slowly adjusts itself to get better. During **inference**, the trained model is given new inputs it has never seen and asked to produce answers. No more studying -- it's test day.

### Parameters: What the Model Learns

**Parameters** are the internal numbers that the model adjusts during training. Think of them as the settings on a mixing board in a recording studio -- each slider (parameter) needs to be in just the right position to produce good sound (good predictions). A small model might have thousands of parameters; GPT-3 has 175 *billion*.

### Traditional Programming vs. ML

| Aspect | Traditional Programming | Machine Learning |
|---|---|---|
| **Input** | Rules + Data | Data + Expected Outputs |
| **Output** | Answers | Learned Rules (a model) |
| **Example** | Spam filter with hand-written rules ("block emails with 'free money'") | Spam filter that learns from 10,000 labeled emails |
| **Adapts?** | Only if a programmer updates the rules | Learns new patterns from new data |
| **Analogy** | Following a recipe | Learning to cook by tasting many dishes |

---

## 2. The Perceptron

The **perceptron** is the simplest possible neural network -- a single artificial neuron. It takes in numbers, multiplies each by a **weight** (how important that input is), adds a **bias** (a baseline adjustment), and passes the result through an **activation function** (a decision rule).

> **Analogy:** Imagine deciding whether to bring an umbrella. You consider inputs like "Are there clouds?" (x1), "Did the forecast say rain?" (x2), and "Is it humid?" (x3). Each factor matters a different amount to you (the weights). The bias is your general tendency ("I usually bring one just in case"). The activation function is your final yes/no decision.

### ASCII Diagram: A Single Perceptron

```
  x₁ --w₁--\
  x₂ --w₂---→ [Σ + b] → [activation] → output
  x₃ --w₃--/
```

The math inside:

$$\text{output} = \text{activation}\left(\sum_{i} w_i x_i + b\right)$$

In plain English: multiply each input by its weight, add them all up, add the bias, then apply the activation function.

### AND Gate: A Worked Example

An AND gate outputs 1 only when *both* inputs are 1. Can a perceptron learn this?

Let's pick weights $w_1 = 1$, $w_2 = 1$, bias $b = -1.5$, and use a step activation (output 1 if the sum is greater than 0, otherwise 0):

| $x_1$ | $x_2$ | $w_1 x_1 + w_2 x_2 + b$ | Output |
|---|---|---|---|
| 0 | 0 | $0 + 0 - 1.5 = -1.5$ | 0 |
| 0 | 1 | $0 + 1 - 1.5 = -0.5$ | 0 |
| 1 | 0 | $1 + 0 - 1.5 = -0.5$ | 0 |
| 1 | 1 | $1 + 1 - 1.5 = 0.5$ | 1 |

It works! The perceptron correctly computes AND.

### The XOR Problem: Why One Neuron Isn't Enough

**XOR** (exclusive or) outputs 1 when the inputs are *different*. Try as you might, no single straight line can separate the XOR outputs on a graph. A single perceptron can only draw one straight line as its decision boundary.

> **Key insight:** This limitation is exactly what motivated researchers to stack multiple perceptrons into layers -- leading us to the next section.

---

## 3. Multi-Layer Perceptron (MLP)

An **MLP** is what you get when you stack perceptrons into layers. The layers between the input and the output are called **hidden layers** -- "hidden" because you don't directly see their values from the outside.

> **Analogy: The Assembly Line.** Imagine a factory where raw materials (inputs) pass through several stations (hidden layers). At each station, workers (neurons) refine the product a little more. The first station might cut raw wood into rough shapes, the second sands them smooth, and the final station paints them. No single station does the whole job, but together they produce a finished product.

### ASCII Diagram: An MLP

```
  Input Layer      Hidden Layer      Output Layer
  ┌───┐            ┌───┐            ┌───┐
  │ x₁│───────────→│ h₁│───────────→│ y₁│
  └───┘    ╲      ╱└───┘╲      ╱    └───┘
            ╲    ╱        ╲    ╱
             ╲  ╱          ╲  ╱
              ╲╱            ╲╱
              ╱╲            ╱╲
             ╱  ╲          ╱  ╲
            ╱    ╲        ╱    ╲
  ┌───┐    ╱      ╲┌───┐╱      ╲    ┌───┐
  │ x₂│───────────→│ h₂│───────────→│ y₂│
  └───┘            └───┘            └───┘
```

Every neuron in one layer connects to every neuron in the next layer. That's why these are also called **fully connected** or **dense** layers.

### Universal Approximation Theorem (Conceptual)

Here's a remarkable fact: with enough neurons in a single hidden layer, an MLP can approximate *any* continuous function. In plain terms, given enough capacity, an MLP can learn any pattern in your data.

> **Caveat:** "Can approximate" doesn't mean "easy to train." In practice, we use many layers (deep networks) rather than one enormous layer, because deep networks learn more efficiently.

### Backpropagation (Conceptual)

**Backpropagation** is how the model learns from its mistakes. After the model makes a prediction:

1. Measure how wrong it was (the **loss**).
2. Work backwards through each layer, figuring out how much each weight contributed to the error.
3. Adjust each weight a little bit to reduce the error.

> **Analogy:** Imagine you bake a cake that's too salty. You trace back: "Was it the frosting? The batter? The specific step where I added salt?" Once you find the culprit, you adjust that step next time. Backpropagation does this automatically for every weight in the network.

---

## 4. Activation Functions

### Why Do We Need Them?

Without activation functions, stacking layers is pointless. A linear function of a linear function is still just a linear function -- you'd get no benefit from depth. Activation functions introduce **non-linearity**, which is what allows neural networks to learn complex, curved, interesting patterns.

> **Analogy:** If every worker on the assembly line could only do one type of operation (say, stretching), the product would never change shape in interesting ways. Activation functions let each layer bend and twist the data into new forms.

### Sigmoid

The **sigmoid** function squashes any number into the range (0, 1). Historically important -- it was the go-to activation for decades.

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

```
  1.0 |                  _______________
      |                /
      |              /
  0.5 |            /
      |          /
      |        /
  0.0 |_______/
      +------|------|------|------|------
            -4    -2      0     2     4
```

### Tanh

**Tanh** is like sigmoid's centered cousin -- it squashes values to (-1, 1) instead of (0, 1). This centered output often helps training.

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

```
  1.0 |                  _______________
      |                /
      |              /
  0.0 |────────────/─────────────────────
      |          /
      |        /
 -1.0 |_______/
      +------|------|------|------|------
            -4    -2      0     2     4
```

### ReLU (Rectified Linear Unit)

**ReLU** is beautifully simple: if the input is positive, keep it; if negative, output zero. It's the most popular activation function in modern networks.

$$\text{ReLU}(x) = \max(0, x)$$

```
      |              /
      |            /
      |          /
      |        /
      |      /
      |    /
  0.0 |___/
      +------|------|------|------|------
            -4    -2      0     2     4
```

### GELU (Gaussian Error Linear Unit)

**GELU** is a smooth version of ReLU -- instead of a hard cutoff at zero, it gently curves. This is the activation function used in GPT models.

$$\text{GELU}(x) = x \cdot \Phi(x)$$

where $\Phi(x)$ is the cumulative distribution function of the standard normal distribution. In simpler terms: it mostly acts like ReLU, but with a smooth transition near zero.

```
      |              /
      |            /
      |          /
      |        /
      |      /
      |    /
  0.0 |__~
      +------|------|------|------|------
            -4    -2      0     2     4
```

*(The `~` represents the gentle curve near zero, unlike ReLU's sharp corner.)*

### Comparison Table

| Function | Range | Shape | Used In | Key Property |
|---|---|---|---|---|
| Sigmoid | (0, 1) | S-curve | Early networks, gates | Smooth, but gradients vanish for large inputs |
| Tanh | (-1, 1) | Centered S-curve | RNNs, LSTMs | Zero-centered, still has vanishing gradient issue |
| ReLU | [0, inf) | Hockey stick | Most modern networks | Simple, fast, but "dead neurons" possible |
| GELU | (~0, inf) | Smooth hockey stick | GPT, BERT, Transformers | Smooth, allows small negative values |

---

## 5. Loss Functions & Optimization

### What Is a Loss Function?

A **loss function** (also called a cost function) measures "how wrong" the model is. A perfect model would have zero loss. During training, the goal is to make the loss as small as possible.

> **Analogy:** A loss function is like the score on a test, except it counts your mistakes. A score of 0 means you got everything right. The model's job is to study (train) until the score is as low as possible.

### Cross-Entropy Loss

**Cross-entropy** is the loss function used for classification tasks (including language modeling). Intuitively, it measures the "surprise" when the model's prediction doesn't match reality.

- If the model says "I'm 99% sure the next word is 'cat'" and it *is* "cat," the loss is tiny (low surprise).
- If the model says "I'm 99% sure the next word is 'cat'" but it's actually "dog," the loss is huge (very surprised!).

$$\text{Cross-Entropy} = -\sum_{i} y_i \log(\hat{y}_i)$$

where $y_i$ is the true label and $\hat{y}_i$ is the model's predicted probability.

### Gradient Descent

> **Analogy: The Blindfolded Hiker.** Imagine you're blindfolded on a hilly landscape, and your goal is to reach the lowest valley. You can't see, but you can feel the slope of the ground under your feet. At each step, you move in the direction that goes most steeply downhill. That's gradient descent.

The **gradient** tells us which direction is "uphill" (increasing loss). We step in the opposite direction (downhill) to decrease the loss.

### Learning Rate: How Big Are Your Steps?

The **learning rate** controls the size of each step during gradient descent.

- **Too large:** You overshoot the valley, bouncing back and forth over it.
- **Too small:** You inch along painfully slowly, and might get stuck in a shallow dip rather than finding the deepest valley.
- **Just right:** You steadily descend into a good minimum.

### Optimizers: SGD and Adam

- **SGD (Stochastic Gradient Descent):** The basic version. Pick a random batch of data, compute the gradient, take a step. Simple but can be slow.
- **Adam (Adaptive Moment Estimation):** A smarter version that adjusts the step size for each parameter individually, based on how the gradients have been behaving recently. It's the most commonly used optimizer for training transformers.

---

## 6. Sequence Models: RNNs

### Why Sequences Matter

Language is inherently sequential -- "The dog bit the man" means something very different from "The man bit the dog." The same words, in different order, have different meanings. Other sequential data includes music, stock prices, and weather patterns.

Standard MLPs treat each input independently -- they have no notion of order or memory. We need a model that can remember what came before.

### Recurrent Neural Networks (RNNs)

An **RNN** processes one element of a sequence at a time, maintaining a **hidden state** -- a kind of memory -- that it passes along to the next step.

> **Analogy:** Reading a book sentence by sentence. After each sentence, you update your mental summary of the story so far. That mental summary is the hidden state.

### ASCII Diagram: An Unrolled RNN

```
  x₁ → [RNN] → h₁ → [RNN] → h₂ → [RNN] → h₃
           ↓              ↓              ↓
          y₁             y₂             y₃
```

At each time step $t$, the RNN takes in the current input $x_t$ and the previous hidden state $h_{t-1}$, and produces a new hidden state $h_t$ and an output $y_t$. The same RNN cell (with the same weights) is reused at every step.

### The Vanishing Gradient Problem

> **Analogy: The Telephone Game.** Remember the game where you whisper a message around a circle, and by the time it gets back to you it's completely garbled? That's what happens to gradients in long sequences. The error signal that backpropagation sends backwards gets weaker and weaker (vanishes) as it passes through many time steps.

This means RNNs struggle to learn connections between words that are far apart. In a long paragraph, an RNN might "forget" what was said at the beginning by the time it reaches the end.

> **This limitation motivated the development of LSTMs and GRUs -- networks specifically designed to remember over long distances.**

---

## 7. LSTMs & GRUs

### LSTM: Long Short-Term Memory

An **LSTM** is an RNN with a special memory mechanism designed to combat the vanishing gradient problem. The key idea is the **cell state** -- a separate channel of information that flows through time, mostly unchanged.

> **Analogy: The Notebook.** Imagine you're taking notes while listening to a long lecture. Your notebook (cell state) carries information forward. You have three tools:
>
> - **Forget gate (eraser):** Decides what old notes to erase because they're no longer relevant.
> - **Input gate (pencil):** Decides what new information to write down.
> - **Output gate (what to share):** Decides which notes to share with others when they ask you a question.

### ASCII Diagram: The Conveyor Belt

The cell state acts like a conveyor belt running along the top of the network. Information rides along it, and the gates selectively add or remove information.

```
  Cell state: ═══════════════════════════════════►
                  ↑ forget    ↑ add new
                  × gate      + gate
                  │           │
  Hidden state: ──┴───────────┴──────────────────►
                                    ↓ output gate
                                    ↓
                                  output
```

The `×` means "multiply" (the forget gate can zero out old info), and the `+` means "add" (the input gate can write in new info). Because addition preserves gradients much better than repeated multiplication, LSTMs can remember information over much longer sequences.

### GRU: Gated Recurrent Unit

A **GRU** is a simplified version of the LSTM. Instead of three gates, it uses two:

- **Reset gate:** How much of the previous hidden state to forget.
- **Update gate:** How much of the new information to blend in.

GRUs are faster to train (fewer parameters) and often perform comparably to LSTMs.

### LSTM vs. GRU Comparison

| Feature | LSTM | GRU |
|---|---|---|
| **Number of gates** | 3 (forget, input, output) | 2 (reset, update) |
| **Separate cell state?** | Yes | No (merged into hidden state) |
| **Parameters** | More | Fewer |
| **Training speed** | Slower | Faster |
| **Performance** | Slightly better on some long-range tasks | Comparable in most cases |
| **When to use** | When you need to capture very long-range dependencies | When training speed matters or data is limited |

---

## 8. The Attention Mechanism

Attention is one of the most important ideas in modern deep learning. It allows a model to focus on the most relevant parts of the input, rather than trying to cram everything into a single fixed-size hidden state.

### The Library Analogy

> Imagine you walk into a library with a **research question** (this is the **Query**, Q).
>
> You look at the **labels and titles** on the spines of books on the shelf (these are the **Keys**, K).
>
> You compare your question to each label: "How relevant is this book to my question?" Books with labels closely matching your question get high relevance scores.
>
> Then you pull books off the shelf and read their **actual contents** (these are the **Values**, V). But you don't read every book equally -- you spend more time on the highly relevant ones and skim or skip the irrelevant ones.
>
> The final result is a weighted blend of the book contents, where the weights come from how well each book's label matched your question.

### The Attention Equation

This is the key equation of modern deep learning:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Breaking it down:

1. $QK^T$ -- Compare the query to every key (dot product measures similarity).
2. $\div \sqrt{d_k}$ -- Scale down so the numbers don't get too large (which would make softmax produce extreme values). Here $d_k$ is the dimension of the keys.
3. $\text{softmax}(\cdot)$ -- Convert similarity scores into probabilities (they sum to 1).
4. $\times V$ -- Use those probabilities to take a weighted average of the values.

### Why Divide by $\sqrt{d_k}$?

When vectors have many dimensions, their dot products tend to grow larger in magnitude. Large inputs to softmax produce outputs that are nearly 0 or 1 (very "peaked"), which makes gradients tiny and learning slow. Dividing by $\sqrt{d_k}$ keeps the values in a gentle range where softmax behaves well.

### Worked Intuitive Example

Consider the sentence: **"The cat sat on the ___"**

When the model predicts the blank, it asks (via attention): "Which earlier words are most relevant to figuring out what comes next?"

- "cat" is highly relevant (it's the subject -- the thing doing the sitting).
- "sat" is relevant (it tells us the action and context).
- "The" and "on" are less informative for this particular prediction.

The attention mechanism automatically assigns higher weights to "cat" and "sat" and lower weights to "The" and "on." This lets the model focus on what matters most.

---

## 9. The Transformer

The **Transformer** architecture (introduced in the 2017 paper "Attention Is All You Need") replaced RNNs as the dominant approach for sequence modeling. Every GPT model is built on it.

### High-Level Idea

> **What makes Transformers special:** Unlike RNNs, which process words one at a time (sequential), Transformers process all words in parallel using attention. This makes them much faster to train and much better at capturing long-range relationships.

### Encoder-Decoder Architecture (Original)

The original Transformer had two halves:

- **Encoder:** Reads the entire input and builds a rich representation of it.
- **Decoder:** Generates the output one token at a time, attending to the encoder's representation.

This was designed for translation (e.g., English in, French out). GPT models simplify this by using only the decoder half (more on this in Section 10).

### Key Components

**Self-Attention:** Every word attends to every other word in the sequence. This is how the model understands context -- "bank" means something different in "river bank" vs. "bank account," and self-attention helps the model figure out which meaning applies.

**Positional Encoding:** Since the Transformer processes all words at once (not sequentially), it has no built-in sense of word order. Positional encodings are added to the input to tell the model "this word is at position 1, this word is at position 2," etc.

**Multi-Head Attention:** Instead of computing attention once, the model computes it multiple times in parallel (each called a "head"), with different learned transformations.

> **Analogy:** Imagine a team of readers analyzing the same document. One reader focuses on grammatical structure, another on sentiment, another on factual content. Each "head" looks at the same words but pays attention to different relationships. Their insights are then combined.

**Residual Connections (Skip Connections):** Each sub-layer has a shortcut that adds the input directly to the output.

> **Analogy:** Like a highway bypass around a small town. The main information can flow straight through, and each layer only needs to learn *what to add* rather than rebuilding everything from scratch.

**Layer Normalization:** Keeps the numbers flowing through the network in a reasonable range, which stabilizes and speeds up training. Think of it as recalibrating your instruments between measurements so they don't drift.

### Simplified Architecture Diagram

```
  Output Probabilities
         ↑
    [Linear + Softmax]
         ↑
  ┌─────────────────┐
  │  Decoder Block   │ ×N
  │  ┌─────────────┐ │
  │  │ Feed-Forward │ │
  │  │   + Add&Norm │ │
  │  ├─────────────┤ │
  │  │Cross-Attention│ │
  │  │   + Add&Norm │ │
  │  ├─────────────┤ │
  │  │Masked Self-  │ │
  │  │  Attention   │ │
  │  │   + Add&Norm │ │
  │  └─────────────┘ │
  └─────────────────┘
         ↑
  [Positional Encoding]
         ↑
    [Embedding]
         ↑
       Input
```

The "×N" means this entire block is repeated N times (stacked). GPT-2 uses N=12 (small) to N=48 (XL). GPT-3 uses N=96.

Each block contains:
1. **Masked self-attention** + Add&Norm
2. **Cross-attention** to the encoder (in the full Transformer; GPT omits this) + Add&Norm
3. **Feed-forward network** + Add&Norm

**What is Add&Norm?** This is shorthand for two operations that happen after every sub-layer (attention or feed-forward):

- **Add (residual connection):** Instead of replacing the input with the sub-layer's output, we *add* the input back to the output: $\text{output} = \text{sublayer}(x) + x$. This creates a "skip connection" — if the sub-layer learns nothing useful, the original signal passes through unchanged. Residual connections are critical for training deep networks because they prevent the gradient from vanishing as it flows backward through many layers.
- **Norm (layer normalization):** After adding, we normalize the values across each token's features so they have a consistent mean and variance. This keeps numbers in a stable range and helps training converge faster. Think of it as recalibrating after each processing step.

**What is the feed-forward network?** This is a small two-layer neural network (an MLP) applied independently to each token's representation. It consists of: Linear layer → activation (GELU in GPT) → Linear layer. The first linear layer typically expands the dimension by 4× (e.g., from 768 to 3072 in GPT-1), and the second projects it back down. While self-attention lets tokens *communicate* with each other, the feed-forward network lets each token *process* that information individually — it is where much of the model's learned "knowledge" is stored.

---

## 10. Decoder-Only Transformers (GPT)

### GPT Uses Only the Decoder

The GPT (Generative Pre-trained Transformer) family uses only the decoder half of the Transformer, with one critical modification: there is no cross-attention to an encoder (since there is no encoder). This simplifies the architecture while retaining the power of self-attention.

### Causal Masking: No Peeking Ahead

In GPT, each word can only attend to words that came *before* it (and itself). This is called **causal masking** -- it prevents the model from "cheating" by looking at future words.

> **Analogy:** It's like reading a mystery novel one page at a time. When you're on page 5, you can reference pages 1-5, but you can't flip ahead to page 10 for the solution.

### ASCII Masking Diagram

This matrix shows which positions can attend to which. A check means "can see," an X means "blocked":

```
       The  cat  sat  on
  The   ✓    ✗    ✗    ✗
  cat   ✓    ✓    ✗    ✗
  sat   ✓    ✓    ✓    ✗
  on    ✓    ✓    ✓    ✓
```

When predicting the word after "sat," the model can attend to "The," "cat," and "sat" -- but not "on" (which hasn't been generated yet).

### Autoregressive Generation

GPT generates text **one token at a time**. After predicting the next word, it appends that word to the input and repeats:

1. Input: "The" -> Predict: "cat"
2. Input: "The cat" -> Predict: "sat"
3. Input: "The cat sat" -> Predict: "on"
4. Input: "The cat sat on" -> Predict: "the"
5. ...and so on.

This is called **autoregressive** generation -- each output becomes part of the input for the next step.

### Connection to Our Reading Papers

| Paper | Model | Key Idea | Week |
|---|---|---|---|
| Radford et al. (2018) | **GPT-1** | Pre-training + fine-tuning on decoder-only Transformer | W3 |
| Radford et al. (2019) | **GPT-2** | Bigger model, zero-shot task performance | W2 |
| Brown et al. (2020) | **GPT-3** | 175B parameters, in-context learning (few-shot) | W1 |
| Ouyang et al. (2022) | **InstructGPT** | RLHF to align GPT-3 with human intent | W4 |

Each paper builds on the same decoder-only Transformer core, scaling it up and refining how it's trained and used.

---

## 11. Fine-Tuning & Transfer Learning

### The Core Idea

> **Analogy: General Education to Medical School.** First, you get a broad education (reading, writing, math, science). Then you specialize: medical school builds on that general foundation. You don't start from scratch -- your general knowledge helps you learn medicine faster and better.

This is exactly what GPT models do:

1. **Pre-training:** Learn general language patterns from an enormous text corpus (books, websites, etc.). This is the "general education" phase.
2. **Fine-tuning:** Adapt to a specific task (like answering questions or summarizing) using a smaller, labeled dataset. This is the "specialization" phase.

### Pre-Training

During pre-training, the model learns to predict the next word in a sequence. Through billions of predictions, it absorbs grammar, facts, reasoning patterns, and even some common sense. The model isn't told what to learn -- it discovers structure on its own.

### Fine-Tuning

Fine-tuning takes the pre-trained model and continues training it on a specific task. Because the model already understands language, it can learn a new task with far fewer examples than training from scratch.

### Few-Shot, One-Shot, and Zero-Shot Learning

GPT-3 (W1) introduced a surprising capability: instead of fine-tuning the model's weights, you can simply show examples in the **prompt**. The model learns "on the fly" from context.

> **Zero-shot:** "Translate this English sentence to French: 'The cat is on the table.'"
> The model receives only an instruction, no examples.

> **One-shot:** "Translate English to French. Example: 'The dog is big' -> 'Le chien est grand.' Now translate: 'The cat is on the table.'"
> The model receives one example.

> **Few-shot:** Same as above but with several examples.

### Comparison Table

| Approach | Examples Provided | Weights Updated? | Data Needed | Performance |
|---|---|---|---|---|
| **Zero-shot** | 0 | No | None | Good for simple tasks |
| **One-shot** | 1 | No | Minimal | Better with an example |
| **Few-shot** | 2-100 (in prompt) | No | Minimal | Strong for many tasks |
| **Fine-tuning** | Hundreds to thousands | Yes | Moderate labeled dataset | Best for specific tasks |

---

## 12. RLHF

> **Note:** This section provides a brief conceptual overview. For the full mathematical treatment, see W4 notes (`notes.md` in `W4_03_17_26/`).

### The Problem

Language models are trained to predict the next word. But "good at predicting the next word" is not the same as "helpful, honest, and harmless." A model trained purely on next-token prediction might:

- Generate toxic or harmful content (because such content exists in the training data).
- Give confident-sounding but wrong answers.
- Refuse to be helpful when it should, or comply when it shouldn't.

> **Analogy:** A student who memorizes the textbook perfectly can ace fill-in-the-blank tests. But that doesn't mean they'll give thoughtful, responsible advice when asked a real-world question. RLHF is how we teach the model to go from "good at fill-in-the-blank" to "actually helpful."

### The 3-Step Process

**RLHF** (Reinforcement Learning from Human Feedback) aligns the model with human preferences through three steps:

```
  Step 1: SFT           Step 2: Reward Model    Step 3: PPO
  ┌──────────┐          ┌──────────┐           ┌──────────┐
  │ Human     │          │ Humans   │           │ RL       │
  │ demos     │ ──────►  │ rank     │ ──────►  │ training │
  │ → fine-   │          │ outputs  │           │ with     │
  │   tune    │          │ → train  │           │ reward   │
  │   GPT-3   │          │   RM     │           │ model    │
  └──────────┘          └──────────┘           └──────────┘
```

**Step 1 -- Supervised Fine-Tuning (SFT):** Human experts write high-quality responses to prompts. The model is fine-tuned on these demonstrations, learning the *style* of a helpful assistant.

**Step 2 -- Reward Model (RM):** The model generates multiple responses to each prompt. Human labelers rank these responses from best to worst. A separate "reward model" is trained to predict which responses humans prefer.

**Step 3 -- Reinforcement Learning (PPO):** The language model generates responses, the reward model scores them, and the language model is updated to produce higher-scoring responses. PPO (Proximal Policy Optimization) is the RL algorithm used to do this without letting the model change too drastically in any single step.

> **For the full mathematical treatment, see W4 notes (`notes.md` in `W4_03_17_26/`).**

---

## Summary: The Building Blocks

> Think of everything in this document as building blocks:
>
> - **Perceptrons** are the bricks -- the smallest computing unit.
> - **MLPs** are the walls -- stacked layers that can learn complex patterns.
> - **Activation functions** are what give the walls their shape -- without them, everything would be flat.
> - **Loss functions & optimizers** are the construction crew -- they measure progress and adjust the building.
> - **RNNs** add the dimension of time -- the building can now process sequences.
> - **LSTMs & GRUs** add better memory -- the building can remember important things from long ago.
> - **Attention** is the architectural plan -- it lets the model focus on what matters most.
> - **The Transformer** is the full house -- the dominant architecture that put it all together.
> - **GPT (decoder-only)** is a specific style of house -- optimized for generating text.
> - **Fine-tuning** is furnishing the house for a specific purpose.
> - **RLHF** is the interior design that makes it livable -- aligning the model with what humans actually want.
>
> You now have the foundation to read and understand the GPT-1 through InstructGPT papers. Welcome to the reading group!
