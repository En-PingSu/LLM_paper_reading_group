# GPT-1: Improving Language Understanding by Generative Pre-Training

**Paper:** Radford et al., OpenAI (2018)

---

## Table of Contents

1. [Challenge](#challenge)
2. [Semi-Supervised Training](#semi-supervised-training)
3. [Architecture](#architecture)
4. [Learning Objective and Loss Function](#learning-objective-and-loss-function)
5. [Pre-Training and Fine-Tuning](#pre-training-and-fine-tuning)
6. [Downstream Tasks](#downstream-tasks)
7. [Why Transformers > LSTMs](#why-transformers--lstms)
8. [Datasets](#datasets)
9. [Glossary](#glossary)

---

## Challenge

Two core open problems in pre-training on unlabelled text:

1. **Which objective to use for pre-training?**
2. **How to transfer learned representations to downstream tasks?**

---

### Pre-Training Objective Candidates

#### 1. Language Modelling (LM)

Learn to predict the next word in a sequence given all previous words.

**Example:**
> "The cat sat on the ___" → model predicts *mat*

**Objective (next-token LM):**

$$\max_{\theta} \sum_{t=1}^{T} \log P_{\theta}(w_t \mid w_1, w_2, \ldots, w_{t-1})$$

**Intuition:** Maximize the probability of the correct next word given all previous words.

---

#### 2. Machine Translation Objective

Learn to map a sentence in one language to its equivalent in another.

**Example:**
> English: "I am hungry" → French: "J'ai faim"

**Objective:**

$$\max_{\theta} \log P_{\theta}(y_1, y_2, \ldots, y_m \mid x_1, x_2, \ldots, x_n)$$

where $x$ is the source sentence and $y$ is the translated sentence.

**Intuition:** Maximize the probability of the correct translation given the source sentence.

---

#### 3. Discourse Coherence Objective

Learn logical relationships between consecutive sentences.

**Example:**
- Sentence A: "John studied all night."
- Sentence B: "He passed the exam."
- Model predicts: B coherently follows A.

**Objective (binary classification):**

$$\max_{\theta} \log P_{\theta}(y \mid s_1, s_2)$$

where $y \in \{0, 1\}$ indicates whether $s_2$ logically follows $s_1$.

**Intuition:** Maximize the probability that sentence pairs are correctly classified as coherent or not.

---

## Semi-Supervised Training

GPT-1 combines **unsupervised pre-training** with **supervised fine-tuning** — a semi-supervised approach for language understanding.

### Evolution of Approaches

#### Word Statistics (Early NLP)

Use large unlabeled text to compute simple statistics (word frequency, co-occurrence) as features for a supervised model.

**Example corpus:**
```
"This movie is amazing"
"Amazing acting and story"
"Terrible movie with bad acting"
```

**Derived statistics:**

| Type | Example |
|------|---------|
| Word frequency | amazing → 2, terrible → 1 |
| Co-occurrence | amazing ↔ movie, terrible ↔ bad |

**Feature vector for "This movie is amazing":**
```
[ amazing=1, terrible=0, movie=1 ]
```

The classifier learns: if "amazing" appears → positive; if "terrible" → negative.

> **Limitation:** Transfers only simple word-level statistics; no understanding of context or deeper meaning.

---

#### Word Embeddings

Learn dense vector representations of words from unlabeled text that capture semantic meaning.

**Example:**
```
amazing → [0.82, 0.11, -0.45]
great   → [0.80, 0.12, -0.43]
```

Trained to predict nearby words from a target word (e.g., Word2Vec, GloVe).

> **Improvement:** Transfers word-level semantic information, but representations are context-independent.

---

#### GPT-1: Contextual Language Representations

Train a language model on large unlabeled text to learn sentence-level, contextual representations, then fine-tune for downstream tasks.

**Example:**
> "The movie was incredibly ___" → predict *good / boring / entertaining* using full sentence context.

> **Improvement:** Transfers higher-level semantics — context and sentence-level meaning.

---

### Progression Summary

| Approach | What is Transferred |
|----------|-------------------|
| Word statistics | Simple frequency/co-occurrence features |
| Word embeddings | Semantic word vectors (context-independent) |
| GPT-1 | Contextual representations via next-token LM |

---

## Architecture

GPT-1 uses a **multi-layer Transformer Decoder** stack.

| Component | Description | Example |
|-----------|-------------|---------|
| **Masked Self-Attention** | Each token attends only to previous tokens (causal masking), preventing the model from seeing future words. | Token "on" in "The cat sat on" can attend to *The, cat, sat* but not to tokens after it. |
| **Layer Normalization** | Normalizes activations within each layer to maintain a consistent value range. | Before attention on "sat", its vector is normalized to prevent exploding/vanishing values. |
| **GELU Activation** | Smoothly gates how strongly features propagate through the network. | Important signals in "cat" are amplified; weaker ones are softly suppressed. |
| **Position-wise FFN** | After attention, each token passes independently through a small network (Linear → GELU → Linear). | "sat" is transformed without interacting with other tokens. |
| **Learned Positional Embeddings** | A learned position vector is added to each token embedding to encode sequence order. | "The cat sat" → $(E(\text{The})+P_0,\ E(\text{cat})+P_1,\ E(\text{sat})+P_2)$ |

![GPT-1 Architecture](Architecture.png)

---

## Learning Objective and Loss Function

### Autoregressive Likelihood

Let a sequence of tokens be $\mathcal{U} = (u_1, u_2, \ldots, u_T)$. By the chain rule of probability:

$$P_{\theta}(\mathcal{U}) = \prod_{t=1}^{T} P_{\theta}(u_t \mid u_1, \ldots, u_{t-1}) \tag{1}$$

### Maximum Likelihood Estimation

Training maximizes the probability of the observed data:

$$\max_{\theta} \prod_{t=1}^{T} P_{\theta}(u_t \mid u_{<t}) \tag{2}$$

### Log-Likelihood

Products of small probabilities are numerically unstable, so we take the log (using $\log(ab) = \log a + \log b$):

$$\max_{\theta} \sum_{t=1}^{T} \log P_{\theta}(u_t \mid u_{<t}) \tag{3}$$

### Negative Log-Likelihood (NLL) Loss

Optimization algorithms minimize a loss, so we negate:

$$\mathcal{L}_{\text{NLL}} = -\sum_{t=1}^{T} \log P_{\theta}(u_t \mid u_{<t}) \tag{4}$$

### Cross-Entropy Equivalence

The model outputs a probability distribution over vocabulary $\mathcal{V}$ via a softmax layer. Let $q_{\theta}(v \mid u_{<t})$ be the predicted probability of token $v$. Define the one-hot target:

$$y_{t,v} = \begin{cases} 1 & v = u_t \\ 0 & \text{otherwise} \end{cases} \tag{5}$$

Equation (4) can then be rewritten as:

$$\mathcal{L} = -\sum_{t=1}^{T} \sum_{v \in \mathcal{V}} y_{t,v} \log q_{\theta}(v \mid u_{<t}) \tag{6}$$

This is exactly the **cross-entropy loss** between the true token distribution $y_t$ and the model prediction $q_{\theta}(\cdot \mid u_{<t})$.

> **Key insight:** Maximizing sequence likelihood, minimizing negative log-likelihood, and minimizing cross-entropy are mathematically equivalent objectives for autoregressive language models.

---

## Pre-Training and Fine-Tuning

Rather than building task-specific architectures, GPT-1 converts structured inputs into a single ordered token sequence, allowing the same pre-trained model to be applied to all tasks.

### Input Transformation Example — Textual Entailment

**Original structure:**
- Sentence A: "A man is playing guitar"
- Sentence B: "A person is playing music"

**Converted sequence:**
```
[START] A man is playing guitar [DELIM] A person is playing music [END]
```

The model then predicts: *entailment / contradiction / neutral*.

> **Fine-tuning note:** An auxiliary language modeling objective is retained during fine-tuning, as unsupervised pre-training already learns several linguistic aspects relevant to target tasks.

---

## Downstream Tasks

### Classification

Predicts a label for a single piece of text.

**Example:** "The movie was amazing" → *positive*

### Similarity

Measures semantic closeness between two sentences.

**Example:** "A dog is running in a field" ↔ "A dog is playing outside" → *high similarity*

### Entailment

Determines whether one sentence logically follows from another.

**Example:** "A man is playing a guitar" → "A person is playing music" → *entailment*

Evaluated on: MNLI-m, MNLI-mm, SNLI, SciTail, QNLI, RTE.

### Multiple Choice

Selects the most plausible answer from several options given a context.

**Example:**
- Context: "The ground was wet and people carried umbrellas."
- Options: "It was raining" / "It was sunny"
- Answer: *It was raining*

---

## Why Transformers > LSTMs

### Heuristic-Based Zero-Shot Reasoning (GPT-1)

Tasks are converted into text sequences, and the model selects the answer with the highest language model log-probability — **no supervised fine-tuning needed**.

- **Sentiment:** Append a prompt word (e.g., "Sentiment: ___") and compare $P(\text{positive})$ vs $P(\text{negative})$.
- **QA:** Choose the answer whose full sequence (document + question + answer) has the highest probability.

**Example (SST-2 Sentiment):**
```
Input:   "The movie was fantastic. Sentiment: ___"
Compare: P(positive) vs P(negative)
Output:  positive  ✓
```

This demonstrates that generative pre-training already learns useful task-relevant knowledge.

### Structural Comparison

| Property | Transformer | LSTM |
|----------|------------|------|
| Token interaction | Self-attention — any token can attend to any other | Sequential — long-range dependencies are harder |
| Parallelism | Fully parallel over sequence | Sequential, harder to parallelize |
| Transfer stability | More stable and stronger transfer | Weaker transfer for long-range tasks |

> **Conclusion:** The attention-based inductive bias in Transformers leads to more stable and better transfer performance across NLP tasks.

---

## Datasets

| Dataset | Task | Description |
|---------|------|-------------|
| **Stories Cloze Test** | Narrative completion | Model selects the correct ending for a short story from two options. |
| **RACE** | Reading comprehension | Multiple-choice questions from English exams for Chinese middle/high school students. |
| **MultiNLI (MNLI)** | Natural language inference | Model classifies sentence pairs as entailment, contradiction, or neutral across multiple genres. |

---

## Glossary

| Term | Definition |
|------|-----------|
| **Generative Pre-Training** | Unsupervised pre-training of a language model on large corpora using next-token prediction. |
| **Discriminative Fine-Tuning** | Supervised adaptation of the pre-trained model to a specific downstream task using labeled data. |
| **Autoregressive LM** | A language model that generates tokens left-to-right, conditioning each token on all previous tokens. |
| **Masked Self-Attention** | Attention mechanism where each position can only attend to positions at or before it in the sequence. |
| **GELU** | Gaussian Error Linear Unit; a smooth activation function used in GPT. |
| **NLL Loss** | Negative log-likelihood loss; equivalent to cross-entropy for one-hot targets. |
| **Chain Rule of Probability** | Decomposes a joint probability into a product of conditional probabilities. |
