# Mathematical Prerequisites for LLM Paper Reading Group

A reference guide covering the math you need to follow along with GPT-1, GPT-2, GPT-3, and InstructGPT. Written for readers ranging from complete ML beginners to those with some ML background. Every formula includes a worked example with step-by-step arithmetic.

---

## Table of Contents

1. [Notation Guide](#1-notation-guide)
2. [Probability](#2-probability)
3. [Information Theory](#3-information-theory)
4. [Linear Algebra](#4-linear-algebra)
5. [Calculus & Optimization](#5-calculus--optimization)
6. [Softmax & Logarithms](#6-softmax--logarithms)
7. [The Sigmoid Function](#7-the-sigmoid-function)
8. [Combinatorics](#8-combinatorics)
9. [Expectation & Summation Notation](#9-expectation--summation-notation)
10. [Connecting Math to Papers](#10-connecting-math-to-papers)

---

## 1. Notation Guide

Mathematical notation is just shorthand. Once you learn the symbols, the papers become much easier to read. Think of each symbol as an abbreviation — the same way "etc." stands for "et cetera."

| Symbol | Name | Meaning | Example |
|--------|------|---------|---------|
| $\theta$ | theta | Model parameters (all the weights and biases) | $L(\theta)$ = loss as a function of model parameters |
| $\phi$ | phi | Alternate set of parameters (e.g., a separate RL policy or reward model) | $\pi_\phi$ = policy parameterized by $\phi$ |
| $\sigma$ | sigma (lowercase) | The sigmoid function, squashes any number into the range (0, 1) | $\sigma(0) = 0.5$ |
| $\pi$ | pi | A policy — a rule that decides what action to take given a situation | $\pi_\theta(a \mid s)$ = probability of action $a$ in state $s$ |
| $\mathbb{E}[\cdot]$ | E | Expectation — the weighted average over all possible outcomes | $\mathbb{E}[X] = \sum x \cdot P(x)$ |
| $\sum$ | sigma (uppercase) | Summation — add up a series of terms | $\sum_{i=1}^{3} i = 1 + 2 + 3 = 6$ |
| $\prod$ | pi (uppercase) | Product — multiply a series of terms | $\prod_{i=1}^{3} i = 1 \times 2 \times 3 = 6$ |
| $\nabla$ | nabla / "del" | Gradient — the vector of all partial derivatives (points "uphill") | $\nabla_\theta L$ = gradient of loss w.r.t. parameters |
| $\arg\max$ | argmax | The input value that produces the maximum output | $\arg\max_x f(x)$ = the $x$ that makes $f$ biggest |
| $\in$ | "in" / "element of" | Membership in a set | $x \in \{1, 2, 3\}$ means $x$ is one of 1, 2, or 3 |
| $\forall$ | "for all" | Universal quantifier — statement holds for every element | $\forall x \in S$ means "for every $x$ in set $S$" |
| $\mid$ | "given" / "such that" | Conditional — what comes after is the condition | $P(A \mid B)$ = probability of $A$ given $B$ |
| $\sim$ | "distributed as" / "sampled from" | A random variable follows a distribution | $x \sim \mathcal{N}(0, 1)$ means $x$ is drawn from a standard normal |
| $\approx$ | "approximately equal" | Two quantities are close but not exactly equal | $\pi \approx 3.14$ |

---

## 2. Probability

Probability is the language of uncertainty. Language models are fundamentally probability machines: they assign probabilities to sequences of words.

### Conditional Probability

**Definition:** The probability of event $A$ happening *given that* event $B$ has already happened.

**Formula:**

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}$$

**Intuition:** Imagine you have a bag of marbles. Conditional probability asks: "If I already know something (B happened), how does that change the chances of something else (A)?"

**Worked Example:**

Suppose we have 100 students. 30 study math, 20 study CS, and 10 study both.

- $P(\text{Math}) = 30/100 = 0.30$
- $P(\text{CS}) = 20/100 = 0.20$
- $P(\text{Math and CS}) = 10/100 = 0.10$

What is the probability a student studies Math, *given* they study CS?

$$P(\text{Math} \mid \text{CS}) = \frac{P(\text{Math} \cap \text{CS})}{P(\text{CS})} = \frac{0.10}{0.20} = 0.50$$

Half of CS students also study math.

> **Key Insight:** In language modeling, we constantly compute conditional probabilities. "What is the probability of the next word, *given* all previous words?" is exactly $P(w_t \mid w_1, w_2, \ldots, w_{t-1})$.

---

### Bayes' Theorem

**Definition:** A way to "reverse" a conditional probability — if you know $P(B \mid A)$, you can find $P(A \mid B)$.

**Formula:**

$$P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}$$

**Intuition:** You see evidence (B) and want to update your belief about a cause (A). Bayes' theorem tells you exactly how to update.

**Worked Example:**

A disease affects 1% of the population. A test is 90% accurate (if you have the disease, 90% chance of positive; if you don't, 90% chance of negative).

You test positive. What is the probability you actually have the disease?

- $P(\text{Disease}) = 0.01$
- $P(\text{Positive} \mid \text{Disease}) = 0.90$
- $P(\text{Positive} \mid \text{No Disease}) = 0.10$
- $P(\text{No Disease}) = 0.99$

First, compute $P(\text{Positive})$ using the law of total probability:

$$P(\text{Positive}) = P(\text{Pos} \mid \text{Disease}) \cdot P(\text{Disease}) + P(\text{Pos} \mid \text{No Disease}) \cdot P(\text{No Disease})$$

$$= 0.90 \times 0.01 + 0.10 \times 0.99 = 0.009 + 0.099 = 0.108$$

Now apply Bayes' theorem:

$$P(\text{Disease} \mid \text{Positive}) = \frac{0.90 \times 0.01}{0.108} = \frac{0.009}{0.108} \approx 0.083$$

Only about 8.3% chance you actually have the disease, even with a positive test! The low base rate (1%) makes false positives dominate.

---

### Joint and Marginal Probability

**Joint probability** $P(A, B)$ is the probability that both $A$ and $B$ happen together.

**Marginal probability** is what you get when you "sum out" one variable from a joint distribution — it is the overall probability of one event regardless of the other.

**Formula (marginalization):**

$$P(A) = \sum_{b} P(A, B = b)$$

**Intuition:** If you have a table of how often combinations occur, the row/column totals are the marginal probabilities.

**Worked Example:**

|  | Sunny | Rainy | **Marginal** |
|--|-------|-------|-------------|
| **Happy** | 0.40 | 0.10 | **0.50** |
| **Sad** | 0.15 | 0.35 | **0.50** |
| **Marginal** | **0.55** | **0.45** | **1.00** |

- Joint: $P(\text{Happy}, \text{Sunny}) = 0.40$
- Marginal: $P(\text{Sunny}) = 0.40 + 0.15 = 0.55$
- Marginal: $P(\text{Happy}) = 0.40 + 0.10 = 0.50$

We "marginalized out" the mood to get the weather probability (or vice versa) by summing across the row or column.

---

### Chain Rule of Probability

**Definition:** Any joint probability can be decomposed into a product of conditional probabilities.

**Formula:**

$$P(w_1, w_2, w_3, \ldots, w_n) = P(w_1) \cdot P(w_2 \mid w_1) \cdot P(w_3 \mid w_1, w_2) \cdots P(w_n \mid w_1, \ldots, w_{n-1})$$

Or more compactly:

$$P(w_1, \ldots, w_n) = \prod_{t=1}^{n} P(w_t \mid w_1, \ldots, w_{t-1})$$

**Intuition:** To compute the probability of a whole sentence, break it into steps: how likely is the first word? Given the first word, how likely is the second? And so on.

**Worked Example:**

Consider the sentence "The cat sat." Suppose:

- $P(\text{The}) = 0.10$
- $P(\text{cat} \mid \text{The}) = 0.05$
- $P(\text{sat} \mid \text{The cat}) = 0.08$

Then:

$$P(\text{The cat sat}) = P(\text{The}) \times P(\text{cat} \mid \text{The}) \times P(\text{sat} \mid \text{The cat})$$

$$= 0.10 \times 0.05 \times 0.08 = 0.0004$$

> **Paper Connection (W3 - GPT-1):** This is exactly the autoregressive language modeling objective. GPT-1 is trained to maximize $\prod_{t=1}^{n} P(w_t \mid w_{t-k}, \ldots, w_{t-1}; \theta)$, which is the chain rule applied to a context window of size $k$. Each transformer layer helps the model compute better estimates of each conditional $P(w_t \mid \text{context})$.

---

## 3. Information Theory

Information theory gives us principled ways to measure "surprise" and compare probability distributions. The loss functions used to train GPT models come directly from information theory.

### Entropy

**Definition:** The average "surprise" or uncertainty in a probability distribution. High entropy means more unpredictable; low entropy means more predictable.

**Formula:**

$$H(X) = -\sum_{x} p(x) \log p(x)$$

(By convention, $0 \cdot \log(0) = 0$. The unit of entropy depends on the log base: $\log$ base 2 gives **bits**, while $\ln$ (base $e$) gives **nats** — short for "natural units." The papers in this reading group all use $\ln$, so entropy values are in nats.)

**Intuition:** Think of a weather forecaster. If the weather is *always* sunny, there is no surprise (low entropy). If it could be anything with equal probability, every day is a surprise (high entropy).

**Worked Example: Fair vs. Biased Coin**

**Fair coin:** $P(\text{H}) = 0.5$, $P(\text{T}) = 0.5$

$$H = -[0.5 \cdot \log_2(0.5) + 0.5 \cdot \log_2(0.5)]$$

$$= -[0.5 \times (-1) + 0.5 \times (-1)]$$

$$= -[-0.5 + (-0.5)]$$

$$= -(-1.0) = 1.0 \text{ bit}$$

**Biased coin:** $P(\text{H}) = 0.9$, $P(\text{T}) = 0.1$

$$H = -[0.9 \cdot \log_2(0.9) + 0.1 \cdot \log_2(0.1)]$$

$$= -[0.9 \times (-0.152) + 0.1 \times (-3.322)]$$

$$= -[-0.137 + (-0.332)]$$

$$= -(-0.469) = 0.469 \text{ bits}$$

The fair coin has higher entropy (1.0 bit) than the biased coin (0.469 bits). This makes sense — the fair coin is harder to predict.

---

### Cross-Entropy

**Definition:** Measures how well a predicted distribution $q$ matches the true distribution $p$. It is always greater than or equal to the entropy of $p$ (with equality when $q = p$).

**Formula:**

$$H(p, q) = -\sum_{x} p(x) \log q(x)$$

**Intuition:** Imagine you are a teacher (the true distribution $p$) and a student is guessing ($q$). Cross-entropy measures how surprised the student is on average. If the student's guesses match reality perfectly, cross-entropy equals entropy. Worse guesses mean higher cross-entropy.

**Worked Example:**

True distribution of next word: $p(\text{cat}) = 0.7$, $p(\text{dog}) = 0.2$, $p(\text{fish}) = 0.1$

Model's predicted distribution: $q(\text{cat}) = 0.5$, $q(\text{dog}) = 0.3$, $q(\text{fish}) = 0.2$

$$H(p, q) = -[0.7 \cdot \ln(0.5) + 0.2 \cdot \ln(0.3) + 0.1 \cdot \ln(0.2)]$$

$$= -[0.7 \times (-0.693) + 0.2 \times (-1.204) + 0.1 \times (-1.609)]$$

$$= -[-0.485 + (-0.241) + (-0.161)]$$

$$= -(-0.887) = 0.887 \text{ nats}$$

Now compare with the entropy of $p$ (the best possible):

$$H(p) = -[0.7 \cdot \ln(0.7) + 0.2 \cdot \ln(0.2) + 0.1 \cdot \ln(0.1)]$$

$$= -[0.7 \times (-0.357) + 0.2 \times (-1.609) + 0.1 \times (-2.303)]$$

$$= -[-0.250 + (-0.322) + (-0.230)]$$

$$= -(-0.802) = 0.802 \text{ nats}$$

Cross-entropy (0.887) > Entropy (0.802). The gap (0.085 nats) is exactly the KL divergence (see next section).

> **Paper Connection (W1 - GPT-3, W3 - GPT-1):** The standard training loss for language models is cross-entropy loss, which is equivalent to negative log-likelihood (NLL). When the model assigns probability $q(w_t)$ to the correct next word $w_t$, the loss for that token is $-\log q(w_t)$. Averaged over all tokens, this is the cross-entropy between the data distribution and the model.

---

### KL Divergence

**Definition:** Measures how different one probability distribution $p$ is from another distribution $q$. It is always non-negative and equals zero only when $p = q$.

**Formula:**

$$D_{KL}(p \| q) = \sum_{x} p(x) \log \frac{p(x)}{q(x)}$$

**Important:** KL divergence is *not* symmetric: $D_{KL}(p \| q) \neq D_{KL}(q \| p)$ in general.

**Intuition:** Think of it as the "extra cost" of using the wrong distribution. If you designed a communication system based on $q$ but the true distribution is $p$, KL divergence measures how many extra bits per message you waste.

**Worked Example:**

Using the same distributions from the cross-entropy example:

- $p$: cat = 0.7, dog = 0.2, fish = 0.1
- $q$: cat = 0.5, dog = 0.3, fish = 0.2

$$D_{KL}(p \| q) = 0.7 \cdot \ln\frac{0.7}{0.5} + 0.2 \cdot \ln\frac{0.2}{0.3} + 0.1 \cdot \ln\frac{0.1}{0.2}$$

$$= 0.7 \cdot \ln(1.4) + 0.2 \cdot \ln(0.667) + 0.1 \cdot \ln(0.5)$$

$$= 0.7 \times 0.336 + 0.2 \times (-0.405) + 0.1 \times (-0.693)$$

$$= 0.235 + (-0.081) + (-0.069)$$

$$= 0.085 \text{ nats}$$

Notice: this equals cross-entropy minus entropy ($0.887 - 0.802 = 0.085$), which is always true by definition: $D_{KL}(p \| q) = H(p, q) - H(p)$.

> **Paper Connection (W4 - InstructGPT, CRITICAL):** The PPO objective in InstructGPT includes a KL penalty term: $-\beta \cdot D_{KL}(\pi_\theta^{RL} \| \pi^{SFT})$. This penalizes the RL-tuned policy $\pi_\theta^{RL}$ for straying too far from the supervised fine-tuned model $\pi^{SFT}$. Without this penalty, the RL policy could "hack" the reward model by producing degenerate outputs that score high but are nonsensical. The coefficient $\beta$ controls how tightly the RL model must stay near the SFT model.

---

## 4. Linear Algebra

Neural networks are, at their core, sequences of matrix multiplications followed by nonlinearities. Linear algebra is the math of matrices and vectors.

### Vectors

**Definition:** A vector is an ordered list of numbers. It can represent a point in space, a direction, or — in ML — a set of features.

**Notation:** Vectors are typically written as bold lowercase letters or with an arrow: $\mathbf{v}$ or $\vec{v}$.

```math
\mathbf{v} = \begin{bmatrix} 3 \\ 1 \\ 4 \end{bmatrix}
```

**Intuition:** Think of a vector as a list of measurements. A word embedding is a vector — each number captures some aspect of the word's meaning. GPT-1 uses 768-dimensional embeddings, meaning each word is represented as a list of 768 numbers.

---

### Dot Products

**Definition:** The dot product of two vectors is the sum of the element-wise products.

**Formula:**

$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i \cdot b_i = a_1 b_1 + a_2 b_2 + \cdots + a_n b_n$$

**Geometric Intuition:** The dot product measures how similar two vectors are. If they point in the same direction, the dot product is large and positive. If they are perpendicular (unrelated), it is zero. If they point in opposite directions, it is large and negative.

**Worked Example:**

```math
\mathbf{a} = \begin{bmatrix} 2 \\ 3 \\ 1 \end{bmatrix}, \quad \mathbf{b} = \begin{bmatrix} 4 \\ 1 \\ 5 \end{bmatrix}
```

$$\mathbf{a} \cdot \mathbf{b} = (2 \times 4) + (3 \times 1) + (1 \times 5) = 8 + 3 + 5 = 16$$

> **Paper Connection (All Weeks):** In the transformer's self-attention mechanism, the attention score between two positions is computed as a dot product of a query vector and a key vector: $\text{score} = \mathbf{q} \cdot \mathbf{k}$. A high dot product means the model thinks those two positions are highly relevant to each other.

---

### Matrix Multiplication

**Definition:** An operation that combines two matrices to produce a new matrix. If $A$ is $m \times n$ and $B$ is $n \times p$, the result $C = AB$ is $m \times p$. Each entry $C_{ij}$ is the dot product of row $i$ of $A$ with column $j$ of $B$.

**Worked Example:**

```math
A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \quad B = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}
```

```math
C = AB = \begin{bmatrix} (1 \times 5 + 2 \times 7) & (1 \times 6 + 2 \times 8) \\ (3 \times 5 + 4 \times 7) & (3 \times 6 + 4 \times 8) \end{bmatrix} = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}
```

Step by step for $C_{11}$: row 1 of $A$ = $[1, 2]$, column 1 of $B$ = $[5, 7]$. Dot product = $1 \times 5 + 2 \times 7 = 5 + 14 = 19$.

> **Paper Connection (All Weeks):** Every linear layer in a transformer performs matrix multiplication: $\text{output} = \mathbf{x} W + \mathbf{b}$. The input vector $\mathbf{x}$ is multiplied by a weight matrix $W$ and a bias $\mathbf{b}$ is added. GPT-3's largest model has 175 billion parameters. The bulk of these are entries in weight matrices, with smaller contributions from bias vectors and embedding tables.

---

### Transpose

**Definition:** The transpose of a matrix $A$, written $A^\top$, swaps rows and columns. If $A$ is $m \times n$, then $A^\top$ is $n \times m$.

**Worked Example:**

```math
A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix} \quad \Rightarrow \quad A^\top = \begin{bmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{bmatrix}
```

Row 1 of $A$ becomes column 1 of $A^\top$, and so on.

---

### Projections (Query, Key, Value)

**Intuition:** A projection takes a vector and "views" it from a particular angle. In the transformer attention mechanism, the same input vector is projected three different ways:

- **Query** ($\mathbf{q} = \mathbf{x} W^Q$): "What am I looking for?"
- **Key** ($\mathbf{k} = \mathbf{x} W^K$): "What do I contain?"
- **Value** ($\mathbf{v} = \mathbf{x} W^V$): "What information do I provide if selected?"

Each projection is a matrix multiplication that transforms the input embedding into a different "view" of the data. The query and key are compared (via dot product) to determine relevance; the value is what actually gets passed along.

**Worked Example:**

Suppose our input embedding is $\mathbf{x} = [1, 0, 2]$ and our query projection matrix is:

```math
W^Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}
```

Then:

```math
\mathbf{q} = \mathbf{x} W^Q = [1, 0, 2] \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix} = [(1 \times 1 + 0 \times 0 + 2 \times 1),\; (1 \times 0 + 0 \times 1 + 2 \times 1)] = [3, 2]
```

The 3-dimensional input has been projected down to a 2-dimensional query vector.

---

## 5. Calculus & Optimization

Training a neural network means finding the parameters that minimize a loss function. Calculus gives us the tools to find that minimum.

### Derivatives

**Definition:** The derivative of a function $f(x)$ at a point tells you the *rate of change* — how much $f$ changes when you nudge $x$ by a tiny amount.

**Notation:** $f'(x)$ or $\frac{df}{dx}$

**Intuition:** Imagine you are hiking and your altitude is $f(x)$ at position $x$. The derivative tells you how steep the slope is. Positive derivative = going uphill. Negative derivative = going downhill. Zero derivative = flat (could be a peak, valley, or saddle point).

**Simple Example:**

If $f(x) = x^2$, then $f'(x) = 2x$.

At $x = 3$: $f'(3) = 2 \times 3 = 6$. The function is increasing at a rate of 6 units per unit change in $x$.

At $x = 0$: $f'(0) = 0$. This is the minimum of $x^2$.

---

### Partial Derivatives

**Definition:** When a function depends on multiple variables, a partial derivative measures the rate of change with respect to *one* variable, holding the others fixed.

**Notation:** $\frac{\partial f}{\partial x}$ (the curly $\partial$ instead of straight $d$ signals "partial").

**Worked Example:**

If $f(x, y) = x^2 + 3xy + y^2$:

$$\frac{\partial f}{\partial x} = 2x + 3y \quad \text{(treat } y \text{ as a constant)}$$

$$\frac{\partial f}{\partial y} = 3x + 2y \quad \text{(treat } x \text{ as a constant)}$$

At $(x, y) = (1, 2)$:

$$\frac{\partial f}{\partial x} = 2(1) + 3(2) = 2 + 6 = 8$$

$$\frac{\partial f}{\partial y} = 3(1) + 2(2) = 3 + 4 = 7$$

The gradient vector $\nabla f = [8, 7]$ points in the direction of steepest increase.

---

### Chain Rule of Calculus

**Definition:** When functions are composed (nested), the chain rule tells you how to compute the derivative of the outer function with respect to the inner variable.

**Formula:** If $y = f(g(x))$, then:

$$\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx} = f'(g(x)) \cdot g'(x)$$

**Important:** This is NOT the probability chain rule from Section 2. Same name, different concepts.

**Worked Example:**

Let $y = (3x + 2)^2$. Here, $f(u) = u^2$ and $g(x) = 3x + 2$.

Step 1: Outer derivative: $\frac{dy}{du} = 2u = 2(3x + 2)$

Step 2: Inner derivative: $\frac{du}{dx} = 3$

Step 3: Multiply: $\frac{dy}{dx} = 2(3x + 2) \times 3 = 6(3x + 2)$

At $x = 1$: $\frac{dy}{dx} = 6(3 \times 1 + 2) = 6 \times 5 = 30$

> **Paper Connection (All Weeks):** Backpropagation — the algorithm that trains neural networks — is just the chain rule applied repeatedly. The loss depends on the output, which depends on the last layer, which depends on the second-to-last layer, and so on. The chain rule lets us compute how each parameter affects the loss by multiplying derivatives along the chain.

---

### Gradient Descent

**Definition:** An optimization algorithm that iteratively adjusts parameters to minimize a loss function, by moving in the direction opposite to the gradient (downhill).

**Formula:**

$$\theta_{\text{new}} = \theta_{\text{old}} - \alpha \cdot \nabla_\theta L$$

Where:
- $\theta$ = model parameters
- $\alpha$ = learning rate (step size)
- $\nabla_\theta L$ = gradient of the loss with respect to parameters
- The minus sign means we go *opposite* to the gradient (downhill, not uphill)

**Intuition:** Imagine you are blindfolded on a hilly landscape and want to reach the lowest point. You feel the slope under your feet (the gradient) and take a step downhill. The learning rate is how big a step you take.

**Worked Example:**

Suppose we have one parameter $\theta = 5$, loss function $L(\theta) = (\theta - 3)^2$, and learning rate $\alpha = 0.1$.

The minimum is at $\theta = 3$ (where $L = 0$).

Step 1: Compute gradient: $\nabla_\theta L = 2(\theta - 3) = 2(5 - 3) = 4$

Step 2: Update: $\theta_{\text{new}} = 5 - 0.1 \times 4 = 5 - 0.4 = 4.6$

Step 3: Repeat. Gradient at 4.6: $2(4.6 - 3) = 3.2$. Update: $4.6 - 0.1 \times 3.2 = 4.6 - 0.32 = 4.28$

Step 4: Gradient at 4.28: $2(4.28 - 3) = 2.56$. Update: $4.28 - 0.256 = 4.024$

The parameter is converging toward the minimum at $\theta = 3$.

---

### Stochastic Gradient Descent (SGD)

**Definition:** Instead of computing the gradient over the *entire* dataset (expensive), SGD estimates the gradient using a small random subset called a **mini-batch**.

**Why it works:** If your mini-batch is randomly sampled, the average gradient over many mini-batches equals the true gradient. Individual estimates are noisy but the noise can actually help escape shallow local minima.

**In practice:** GPT-3 uses batch sizes of up to 3.2 million tokens. Even that is a tiny fraction of the 300 billion tokens in the training data.

---

### Learning Rate Schedules

**Definition:** Instead of using a fixed learning rate $\alpha$, we change it over the course of training.

**Cosine Decay:**

The learning rate follows a cosine curve, starting high and smoothly decreasing to near zero:

$$\alpha_t = \alpha_{\min} + \frac{1}{2}(\alpha_{\max} - \alpha_{\min})\left(1 + \cos\left(\frac{t}{T} \cdot \pi\right)\right)$$

Where $t$ is the current step and $T$ is the total number of steps.

**Worked Example:**

Suppose $\alpha_{\max} = 0.001$, $\alpha_{\min} = 0$, $T = 100$ steps.

At $t = 0$ (start):

$$\alpha_0 = 0 + \frac{1}{2}(0.001)\left(1 + \cos(0)\right) = 0.0005 \times (1 + 1) = 0.001$$

At $t = 50$ (halfway):

$$\alpha_{50} = 0.0005 \times \left(1 + \cos(0.5\pi)\right) = 0.0005 \times (1 + 0) = 0.0005$$

At $t = 100$ (end):

$$\alpha_{100} = 0.0005 \times \left(1 + \cos(\pi)\right) = 0.0005 \times (1 - 1) = 0$$

The learning rate smoothly decreases from 0.001 to 0.

> **Paper Connection (W4 - InstructGPT):** The InstructGPT paper uses cosine learning rate decay for SFT (supervised fine-tuning) training over 16 epochs. Starting with a larger learning rate helps the model learn quickly; decaying it helps the model settle into a good minimum without overshooting.

---

## 6. Softmax & Logarithms

### Logarithm Rules

Logarithms turn multiplication into addition, which makes many calculations much more manageable.

| Rule | Formula | Example |
|------|---------|---------|
| Product rule | $\log(a \cdot b) = \log(a) + \log(b)$ | $\log(2 \times 8) = \log(2) + \log(8) = 0.301 + 0.903 = 1.204 = \log(16)$ |
| Quotient rule | $\log(a / b) = \log(a) - \log(b)$ | $\log(8/2) = \log(8) - \log(2) = 0.903 - 0.301 = 0.602 = \log(4)$ |
| Power rule | $\log(a^n) = n \cdot \log(a)$ | $\log(2^3) = 3 \cdot \log(2) = 3 \times 0.301 = 0.903 = \log(8)$ |

(Examples above use $\log_{10}$.)

---

### Why Logs? Numerical Stability

**The problem:** When computing the probability of a sentence, we multiply many small probabilities together:

$$P(\text{sentence}) = 0.1 \times 0.05 \times 0.08 \times 0.03 \times \ldots$$

After enough multiplications, this number becomes astronomically tiny (like $10^{-50}$), causing **underflow** — the computer rounds it to zero.

**The solution:** Work in log-space. Take the log of the probability:

$$\log P(\text{sentence}) = \log(0.1) + \log(0.05) + \log(0.08) + \log(0.03) + \ldots$$

$$= -1.0 + (-1.301) + (-1.097) + (-1.523) + \ldots$$

These are ordinary-sized numbers that computers handle easily.

> **Paper Connection:** This is why language models are trained with **negative log-likelihood** (NLL) loss rather than directly maximizing likelihood. Minimizing $-\log P$ is equivalent to maximizing $P$, but numerically stable.

---

### The Exponential Function

**Definition:** $\exp(x) = e^x$ where $e \approx 2.718$.

The exponential function is the inverse of the natural logarithm: $\exp(\ln(x)) = x$ and $\ln(\exp(x)) = x$.

Key values: $\exp(0) = 1$, $\exp(1) \approx 2.718$, $\exp(-1) \approx 0.368$.

---

### Softmax

**Definition:** A function that converts a vector of arbitrary real numbers (called **logits**) into a probability distribution (all positive, sums to 1).

**Formula:**

$$\text{softmax}(z_i) = \frac{\exp(z_i)}{\sum_{j=1}^{K} \exp(z_j)}$$

**Intuition:** Each logit gets "exponentiated" to make it positive, then we divide by the total to normalize. Larger logits get exponentially larger probabilities, so softmax amplifies differences.

**Worked Example with 3 logits:**

Suppose the model outputs logits $\mathbf{z} = [2.0, 1.0, 0.5]$ for three vocabulary words.

Step 1: Exponentiate each:

$$\exp(2.0) = 7.389, \quad \exp(1.0) = 2.718, \quad \exp(0.5) = 1.649$$

Step 2: Sum the exponentials:

$$7.389 + 2.718 + 1.649 = 11.756$$

Step 3: Divide each by the sum:

$$\text{softmax}(2.0) = \frac{7.389}{11.756} = 0.629$$

$$\text{softmax}(1.0) = \frac{2.718}{11.756} = 0.231$$

$$\text{softmax}(0.5) = \frac{1.649}{11.756} = 0.140$$

Result: $[0.629, 0.231, 0.140]$

Check: $0.629 + 0.231 + 0.140 = 1.000$. It is a valid probability distribution.

> **Paper Connection (All Weeks):** The final layer of every GPT model applies softmax over the vocabulary. If GPT-2's vocabulary has 50,257 tokens, the model produces 50,257 logits, and softmax converts them into 50,257 probabilities — one for each possible next token.

---

## 7. The Sigmoid Function

### Formula

$$\sigma(x) = \frac{1}{1 + \exp(-x)}$$

### Properties

| Property | Value |
|----------|-------|
| Output range | $(0, 1)$ — always between 0 and 1 but never exactly 0 or 1 |
| $\sigma(0)$ | $0.5$ (the midpoint) |
| Symmetry | $\sigma(-x) = 1 - \sigma(x)$ |
| Large positive $x$ | $\sigma(x) \to 1$ |
| Large negative $x$ | $\sigma(x) \to 0$ |

**Intuition:** The sigmoid function is a "squashing" function. It takes any real number and maps it to a probability-like value between 0 and 1. It is smooth, S-shaped, and centered at 0.5.

### Worked Example

Compute $\sigma(x)$ for several values:

**At $x = 0$:**

$$\sigma(0) = \frac{1}{1 + \exp(0)} = \frac{1}{1 + 1} = \frac{1}{2} = 0.5$$

**At $x = 2$:**

$$\sigma(2) = \frac{1}{1 + \exp(-2)} = \frac{1}{1 + 0.135} = \frac{1}{1.135} = 0.881$$

**At $x = -2$:**

$$\sigma(-2) = \frac{1}{1 + \exp(2)} = \frac{1}{1 + 7.389} = \frac{1}{8.389} = 0.119$$

Notice: $\sigma(2) + \sigma(-2) = 0.881 + 0.119 = 1.0$. This confirms the symmetry property.

**At $x = 5$:**

$$\sigma(5) = \frac{1}{1 + \exp(-5)} = \frac{1}{1 + 0.0067} = \frac{1}{1.0067} = 0.993$$

Already very close to 1.

> **Paper Connection (W4 - InstructGPT):** The reward model loss function uses sigmoid to convert a score difference into a preference probability:
>
> $$\text{loss} = -\log\left(\sigma(r_w - r_l)\right)$$
>
> Here, $r_w$ is the reward model's score for the *preferred* (winning) response and $r_l$ is the score for the *rejected* (losing) response. The term $\sigma(r_w - r_l)$ represents the probability that the reward model correctly ranks the winning response higher. By taking $-\log$ of this, we create a loss that is small when $r_w \gg r_l$ (correct ranking, high confidence) and large when $r_w \leq r_l$ (wrong ranking).
>
> **Numerical example:** If $r_w = 3.0$ and $r_l = 1.0$, then $\sigma(3.0 - 1.0) = \sigma(2.0) = 0.881$, so loss $= -\log(0.881) = 0.127$. Small loss — the model correctly ranks the preferred response much higher. If instead $r_w = 1.5$ and $r_l = 1.4$, then $\sigma(0.1) = 0.525$, so loss $= -\log(0.525) = 0.644$. Larger loss — the model barely distinguishes the two.

---

## 8. Combinatorics

### Factorial Notation

**Definition:** $n!$ (read "$n$ factorial") is the product of all positive integers from 1 to $n$.

$$n! = n \times (n-1) \times (n-2) \times \cdots \times 2 \times 1$$

By convention, $0! = 1$.

**Examples:**
- $3! = 3 \times 2 \times 1 = 6$
- $4! = 4 \times 3 \times 2 \times 1 = 24$
- $5! = 120$

---

### Binomial Coefficient

**Definition:** $\binom{n}{k}$ (read "$n$ choose $k$") counts the number of ways to choose $k$ items from $n$ items, where order does not matter.

**Formula:**

$$\binom{n}{k} = \frac{n!}{k!(n-k)!}$$

**Intuition:** If you have $n$ objects and want to pick $k$ of them (without caring about the order you pick them), this formula tells you how many distinct groups you can form.

### Worked Example: C(4, 2) = 6

$$\binom{4}{2} = \frac{4!}{2! \cdot 2!} = \frac{4 \times 3 \times 2 \times 1}{(2 \times 1)(2 \times 1)} = \frac{24}{2 \times 2} = \frac{24}{4} = 6$$

The 6 pairs from items {A, B, C, D}: {A,B}, {A,C}, {A,D}, {B,C}, {B,D}, {C,D}.

> **Paper Connection (W4 - InstructGPT):** This is exactly what InstructGPT uses! When human labelers rank $K$ responses to a prompt, the authors generate all $\binom{K}{2}$ pairwise comparisons for training the reward model. With $K = 4$ responses, $\binom{4}{2} = 6$ comparison pairs are created from a single ranking. This is far more data-efficient than collecting each pairwise comparison independently.

### Worked Example: C(9, 2) = 36

$$\binom{9}{2} = \frac{9!}{2! \cdot 7!} = \frac{9 \times 8}{2 \times 1} = \frac{72}{2} = 36$$

> The InstructGPT paper uses $K$ values ranging from 4 to 9. With $K = 9$, a single ranking produces 36 pairwise comparisons — a huge multiplier on the labeling effort.

---

## 9. Expectation & Summation Notation

### Summation ($\sum$)

**Definition:** The summation symbol $\sum$ means "add up a series of terms."

**Notation:**

$$\sum_{i=1}^{n} a_i = a_1 + a_2 + a_3 + \cdots + a_n$$

The variable $i$ starts at the bottom value (1) and increments to the top value ($n$).

**Worked Example:**

$$\sum_{i=1}^{4} i^2 = 1^2 + 2^2 + 3^2 + 4^2 = 1 + 4 + 9 + 16 = 30$$

---

### Product Notation ($\prod$)

**Definition:** The product symbol $\prod$ means "multiply a series of terms."

**Notation:**

$$\prod_{i=1}^{n} a_i = a_1 \times a_2 \times a_3 \times \cdots \times a_n$$

**Worked Example:**

$$\prod_{i=1}^{4} i = 1 \times 2 \times 3 \times 4 = 24 \quad (= 4!)$$

> **Paper Connection:** The likelihood of a sequence under a language model is a product: $\prod_{t=1}^{T} P(w_t \mid w_{<t})$. This is why we often switch to log-likelihood (turning the product into a sum) for computational convenience.

---

### Expected Value

**Definition:** The expected value $\mathbb{E}[X]$ of a random variable $X$ is its weighted average, where each outcome is weighted by its probability.

**Formula:**

$$\mathbb{E}[X] = \sum_{x} x \cdot P(x)$$

**Intuition:** If you repeated an experiment infinitely many times, the expected value is the average result you would observe. It is the "center of gravity" of the distribution.

**Worked Example:**

Suppose you roll a loaded die with these probabilities:

| Face ($x$) | 1 | 2 | 3 | 4 | 5 | 6 |
|------------|-----|-----|-----|-----|-----|-----|
| $P(x)$ | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.5 |

$$\mathbb{E}[X] = 1(0.1) + 2(0.1) + 3(0.1) + 4(0.1) + 5(0.1) + 6(0.5)$$

$$= 0.1 + 0.2 + 0.3 + 0.4 + 0.5 + 3.0 = 4.5$$

The expected value is 4.5 — higher than a fair die's expected value of 3.5, because this die is loaded toward 6.

---

### Expectation Notation in the InstructGPT Paper

In the InstructGPT paper, you will encounter expressions like:

$$\mathbb{E}_{(x, y_w, y_l) \sim D}\left[-\log\left(\sigma(r_\theta(x, y_w) - r_\theta(x, y_l))\right)\right]$$

Let us decode this piece by piece:

| Notation | Meaning |
|----------|---------|
| $\mathbb{E}$ | "Take the average of..." |
| $(x, y_w, y_l)$ | A triple: prompt $x$, winning response $y_w$, losing response $y_l$ |
| $\sim D$ | "...where the triples are drawn from dataset $D$" |
| $r_\theta(x, y_w)$ | Reward model's score for the prompt-response pair $(x, y_w)$ |
| $\sigma(\cdot)$ | Sigmoid function, converting score difference to probability |
| $-\log(\cdot)$ | Negative log, turning probability into loss |

**In plain English:** "Average the loss $-\log(\sigma(r_w - r_l))$ over all (prompt, preferred response, rejected response) triples in the dataset."

**Worked Example:**

Suppose the dataset $D$ has 3 examples:

| Example | $r_w$ | $r_l$ | $r_w - r_l$ | $\sigma(\cdot)$ | $-\log(\sigma(\cdot))$ |
|---------|--------|--------|-------------|-----------------|----------------------|
| 1 | 4.0 | 1.0 | 3.0 | 0.953 | 0.048 |
| 2 | 2.5 | 2.0 | 0.5 | 0.622 | 0.475 |
| 3 | 1.0 | 1.5 | -0.5 | 0.378 | 0.972 |

The expected loss:

$$\mathbb{E}[\text{loss}] = \frac{1}{3}(0.048 + 0.475 + 0.972) = \frac{1.495}{3} = 0.498$$

Notice that Example 3 has $r_w < r_l$ (the model scored the rejected response higher than the preferred one), producing the largest loss. Training will push the model to fix this.

---

## 10. Connecting Math to Papers

This table maps each mathematical concept to the weeks where it appears. Use it as a quick reference when reading the papers.

| Concept | W1 (GPT-3) | W2 (GPT-2) | W3 (GPT-1) | W4 (InstructGPT) |
|---------|------------|------------|------------|-------------------|
| **Chain rule of probability** | Autoregressive LM objective | Zero-shot task formulation | Core training objective $\prod P(w_t \mid w_{t-k}, \ldots, w_{t-1})$ | Language model pre-training baseline |
| **Cross-entropy / NLL** | Training loss; reported in bits-per-byte | Perplexity evaluation | Pre-training loss $L_1(\mathcal{U})$ | SFT training loss |
| **KL divergence** | — | — | — | PPO penalty $\beta \cdot D_{KL}(\pi_\theta^{RL} \| \pi^{SFT})$ |
| **Sigmoid ($\sigma$)** | — | — | — | RM loss: $-\log(\sigma(r_w - r_l))$ |
| **Binomial coefficients** | — | — | — | RM pairwise comparisons: $\binom{K}{2}$ pairs from $K$ ranked responses |
| **Softmax** | Output layer over vocabulary | Output layer over vocabulary | Output layer over vocabulary | Output layer over vocabulary |
| **Gradient descent / SGD** | Training with Adam optimizer | Training with Adam optimizer | Training with Adam optimizer | Training RM, SFT, and PPO models |
| **Cosine LR decay** | Cosine schedule during training | — | — | SFT training: 16 epochs with cosine LR decay |
| **Expectation ($\mathbb{E}$)** | — | — | — | PPO objective: $\mathbb{E}[r_\theta - \beta D_{KL}]$ |
| **Matrix multiplication** | 175B parameter weight matrices | Weight matrices in 48-layer transformer | Weight matrices in 12-layer transformer | Same architecture, fine-tuned |
| **Dot product** | Attention score computation | Attention score computation | Attention score computation | Attention score computation |
| **Bayes' theorem** | Conceptual basis for probabilistic modeling | Conceptual basis | Conceptual basis | — |
| **Entropy** | Reported via perplexity and BPB | Reported via perplexity | — | — |
| **Log rules** | Numerical stability in training | Numerical stability in training | Log-likelihood objective | Log in RM loss and PPO objective |

> **Reading tip:** If you encounter unfamiliar math in a paper, use this table to find the relevant section above, work through the example, then return to the paper. Most of the heavy math is concentrated in W4 (InstructGPT), which combines sigmoid, KL divergence, expectation, and combinatorics in a single training pipeline.
