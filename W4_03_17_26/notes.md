# Week 4 — Paper Notes
**Paper:** Training Language Models to Follow Instructions with Human Feedback (InstructGPT), Ouyang et al. 2022

---

## Table of Contents

1. [Overview](#overview)
2. [Things That Came Up During Reading](#things-that-came-up-during-reading)
3. [Key Points](#key-points)
4. [The Alignment Problem](#the-alignment-problem)
5. [The RLHF Pipeline](#the-rlhf-pipeline)
   - [Step 1: Supervised Fine-Tuning (SFT)](#step-1-supervised-fine-tuning-sft)
   - [Step 2: Reward Model (RM)](#step-2-reward-model-rm)
   - [Step 3: Proximal Policy Optimization (PPO)](#step-3-proximal-policy-optimization-ppo)
6. [PPO-ptx: Mitigating the Alignment Tax](#ppo-ptx-mitigating-the-alignment-tax)
7. [Key Results](#key-results)
8. [Evaluation Framework](#evaluation-framework)
9. ["Who Are We Aligning To?"](#who-are-we-aligning-to)
10. [Datasets & Benchmarks](#datasets--benchmarks)
11. [Glossary](#glossary)

---

## Overview
*Paper reference: Abstract & Section 1 (pp. 1–3)*

The language modeling objective — predict the next token on web text — is fundamentally different from what users actually want. Making models bigger does not fix this: a 175B GPT-3 can still be untruthful, toxic, or unhelpful. InstructGPT demonstrates that reinforcement learning from human feedback (RLHF) can align language models with user intent. A 1.3B InstructGPT model is preferred over a 175B GPT-3 despite having 100x fewer parameters.

---

## Things That Came Up During Reading

> *(Add specific observations, confusions, and aha moments here as you read.)*

---

## Key Points
*Paper reference: Section 1 (pp. 1–4)*

- The language modeling objective (next-token prediction on internet text) is misaligned with user intent (helpful, honest, harmless)
- RLHF provides a three-stage pipeline (SFT → Reward Model → PPO) to align model behavior with human preferences
- A 1.3B parameter InstructGPT model is preferred by humans over a 175B GPT-3, showing alignment is more important than scale
- The reward model is trained on 33k human comparison rankings using pairwise preferences, not absolute scores
- A KL divergence penalty prevents the RL policy from drifting too far from the SFT baseline (reward hacking)
- RLHF introduces an "alignment tax" — performance regression on standard NLP benchmarks — which PPO-ptx mitigates by mixing pretraining gradients
- The paper acknowledges that alignment is to a narrow group of labelers, not to humanity broadly

---

## The Alignment Problem
*Paper reference: Section 1 (pp. 1–2)*

| Aspect | LM Optimization | User Expectations |
|--------|----------------|-------------------|
| **Objective** | Predict next token from web text | Follow instructions, be helpful |
| **Training data** | Raw internet text (all of it, including toxic/false content) | High-quality, curated demonstrations |
| **Truthfulness** | Reproduces statistical patterns (including falsehoods) | Statements should be factually accurate |
| **Safety** | No safety constraint; toxic text is valid training signal | Should refuse harmful requests, avoid toxic output |
| **Output format** | Mimics distribution of training data | Structured, concise, task-appropriate responses |
| **Success metric** | Low perplexity on held-out web text | Helpful, honest, and harmless to the user |

---

## The RLHF Pipeline
*Paper reference: Section 3 — Methods and experimental details (pp. 6–9)*

### Step 1: Supervised Fine-Tuning (SFT)
*Paper reference: Section 3.5 "Supervised fine-tuning (SFT)" (p. 8)*

The first stage collects human-written demonstrations of ideal model behavior and fine-tunes GPT-3 on them.

**Training details:**
- ~13k demonstrations (labeler-written prompts + prompts from the OpenAI API)
- 16 epochs, cosine learning rate decay, residual dropout 0.2
- Overfits on validation loss after 1 epoch, but more training epochs improve RM score and human preference ratings
- Model selection via RM score on the validation set (not validation loss)

**Worked example — what a demonstration looks like:**

- **Prompt:** "Explain the moon landing to a 6-year-old."
- **Demonstration:** "Some people went on a really big rocket ship to the moon. They wore special suits so they could breathe, and when they got there, they walked around and even planted a flag! Then they flew all the way back home to Earth."

The SFT model learns to produce this style of helpful, instruction-following response rather than simply continuing the text as a web document.

---

### Step 2: Reward Model (RM)
*Paper reference: Section 3.5 "Reward modeling (RM)" (p. 8, Equation 1)*

The reward model learns to assign scalar scores to model outputs based on human preference rankings.

**Architecture and training:**
- 6B parameters only (the 175B RM was unstable during training)
- Starts from the **SFT model checkpoint** — this is the saved set of model weights from the end of Step 1 (supervised fine-tuning). A checkpoint is a snapshot of all the model's learned parameters at a given point in training. Rather than training the reward model from scratch, the authors initialize it with these SFT weights so it already "understands" language. The weights are not frozen; they continue to be updated during reward model training.
- The **final unembedding layer is removed**. In a normal language model, the last layer (the "unembedding" or "language model head") is a matrix that projects the model's internal representation (a vector of e.g. 4096 numbers) into a probability distribution over the entire vocabulary (e.g. 50,257 tokens). Since the reward model does not need to predict next tokens, this layer is replaced by a simple linear projection that maps the internal representation down to a **single scalar number** — the reward score.
- Trained on 33k comparison rankings
- For each prompt, $K = 4$ to $K = 9$ responses are ranked by labelers, producing $\binom{K}{2}$ pairwise comparisons
- All $\binom{K}{2}$ pairs from the same prompt are trained as a single batch item (prevents overfitting and is efficient — only one forward pass per completion)

**Loss function:**

$$\text{loss}(\theta) = -\frac{1}{\binom{K}{2}} E_{(x, y_w, y_l) \sim D}\left[\log\left(\sigma\left(r_\theta(x, y_w) - r_\theta(x, y_l)\right)\right)\right]$$

**Component-by-component breakdown:**

| Symbol | Meaning |
|--------|---------|
| $r_\theta(x, y)$ | Scalar reward score output by the RM for prompt $x$ and completion $y$ |
| $y_w$ | The preferred (winning) completion |
| $y_l$ | The less preferred (losing) completion |
| $\sigma$ | Sigmoid function, converts score difference to a probability in $(0, 1)$ |
| $\log(\sigma(\cdot))$ | Log-probability — we maximize this (equivalently, minimize the negative) |
| $\frac{1}{\binom{K}{2}}$ | Normalization by the number of pairs per prompt |
| $E_{(x, y_w, y_l) \sim D}$ | Expectation (average) over all comparison triples in the dataset |

**Intuition:** If the RM correctly assigns a higher score to the preferred completion, then $r_\theta(x, y_w) - r_\theta(x, y_l) > 0$, the sigmoid is above 0.5, and the log is closer to 0 (low loss). If the RM gets it wrong, the difference is negative, the sigmoid is below 0.5, and the log is a large negative number (high loss).

---

**Worked example ($K = 4$):**

Given a prompt $x$, a labeler ranks 4 responses: $A \succ B \succ C \succ D$.

**Step 1 — Count pairs:**

$$\binom{4}{2} = \frac{4!}{2! \cdot 2!} = \frac{24}{4} = 6 \text{ pairs}$$

**Step 2 — List all 6 pairs** (winner, loser):

| Pair | $y_w$ | $y_l$ |
|------|--------|--------|
| 1 | A | B |
| 2 | A | C |
| 3 | A | D |
| 4 | B | C |
| 5 | B | D |
| 6 | C | D |

**Step 3 — Suppose the RM assigns these scores:**

| Response | $r_\theta$ |
|----------|-----------|
| A | 2.0 |
| B | 1.0 |
| C | 0.5 |
| D | -0.5 |

**Step 4 — Compute $\sigma(r_w - r_l)$ and $\log(\sigma(r_w - r_l))$ for each pair:**

Recall $\sigma(z) = \frac{1}{1 + e^{-z}}$.

| Pair | $r_w - r_l$ | $\sigma(\cdot)$ | $\log(\sigma(\cdot))$ |
|------|-------------|-----------------|----------------------|
| (A, B) | $2.0 - 1.0 = 1.0$ | $\frac{1}{1+e^{-1.0}} = \frac{1}{1+0.368} = 0.731$ | $\log(0.731) = -0.313$ |
| (A, C) | $2.0 - 0.5 = 1.5$ | $\frac{1}{1+e^{-1.5}} = \frac{1}{1+0.223} = 0.818$ | $\log(0.818) = -0.201$ |
| (A, D) | $2.0 - (-0.5) = 2.5$ | $\frac{1}{1+e^{-2.5}} = \frac{1}{1+0.082} = 0.924$ | $\log(0.924) = -0.079$ |
| (B, C) | $1.0 - 0.5 = 0.5$ | $\frac{1}{1+e^{-0.5}} = \frac{1}{1+0.607} = 0.622$ | $\log(0.622) = -0.475$ |
| (B, D) | $1.0 - (-0.5) = 1.5$ | $\frac{1}{1+e^{-1.5}} = \frac{1}{1+0.223} = 0.818$ | $\log(0.818) = -0.201$ |
| (C, D) | $0.5 - (-0.5) = 1.0$ | $\frac{1}{1+e^{-1.0}} = \frac{1}{1+0.368} = 0.731$ | $\log(0.731) = -0.313$ |

**Step 5 — Average the log-sigmoid values:**

$$\text{average} = \frac{(-0.313) + (-0.201) + (-0.079) + (-0.475) + (-0.201) + (-0.313)}{6} = \frac{-1.582}{6} = -0.264$$

**Step 6 — Negate to get the loss:**

$$\text{loss} = -(-0.264) = 0.264$$

> **Interpretation:** A loss of 0.264 is relatively low, meaning the RM's scores are well-calibrated with the human ranking. If the RM had scored $D > C > B > A$ (completely reversed), the loss would be much higher. A perfect RM with infinite score gaps would drive the loss toward 0.

---

### Step 3: Proximal Policy Optimization (PPO)
*Paper reference: Section 3.5 "Reinforcement learning (RL)" (p. 9, Equation 2)*

The final stage uses the trained reward model to optimize the language model's policy via reinforcement learning.

**Objective:**

$$\text{objective}(\phi) = E_{(x,y) \sim D_{\pi^{RL}_\phi}}\left[r_\theta(x, y) - \beta \log \frac{\pi^{RL}_\phi(y \mid x)}{\pi^{SFT}(y \mid x)}\right] + \gamma E_{x \sim D_{\text{pretrain}}}\left[\log(\pi^{RL}_\phi(x))\right]$$

**Component-by-component explanation:**

| Component | Meaning |
|-----------|---------|
| $r_\theta(x, y)$ | Reward from the trained RM for prompt $x$ and response $y$ |
| $\beta \log \frac{\pi^{RL}_\phi(y \mid x)}{\pi^{SFT}(y \mid x)}$ | KL divergence penalty — prevents the RL policy from drifting too far from the SFT model, guarding against reward hacking |
| $\gamma E_{x \sim D_{\text{pretrain}}}[\log(\pi^{RL}_\phi(x))]$ | Pretraining mix term — maintains language modeling performance on standard NLP benchmarks |
| $\gamma = 0$ | "PPO" models (no pretraining mix) |
| $\gamma > 0$ | "PPO-ptx" models (includes pretraining mix) |

**Training environment:**
- Bandit setting — a prompt $x$ is sampled, the model generates a full response $y$, receives a reward $r_\theta(x, y)$, and the episode ends
- The value function is initialized from the RM
- A per-token KL penalty from the SFT model is applied during generation to keep the policy stable

**Worked example:**

Suppose:
- Prompt $x$: "What is the capital of France?"
- The RL policy generates $y$: "The capital of France is Paris."
- $r_\theta(x, y) = 3.2$ (high reward from RM)
- $\log \frac{\pi^{RL}_\phi(y \mid x)}{\pi^{SFT}(y \mid x)} = 0.8$ (the RL policy has drifted somewhat from SFT)
- $\beta = 0.1$

Then the per-example objective contribution is:

$$3.2 - 0.1 \times 0.8 = 3.2 - 0.08 = 3.12$$

If the RL policy drifted much further, say $\log \frac{\pi^{RL}}{\pi^{SFT}} = 5.0$:

$$3.2 - 0.1 \times 5.0 = 3.2 - 0.5 = 2.7$$

The KL penalty increasingly reduces the objective as the policy diverges, discouraging reward hacking.

---

## PPO-ptx: Mitigating the Alignment Tax
*Paper reference: Section 3.5 (p. 9) and Section 4.2 (pp. 14–15, Figures 29, 33–34 in Appendix)*

**Definition:** The alignment tax is the performance regression on standard NLP benchmarks caused by RLHF fine-tuning. The model becomes better at following instructions but worse at general language tasks.

**Mechanism:** PPO-ptx mixes pretraining gradients into PPO training via the $\gamma$ term in the PPO objective. At each training step, the model receives both RL gradients (from the reward model) and language modeling gradients (from pretraining data).

**Benchmark comparison:**

| Benchmark | GPT-3 | PPO | PPO-ptx | Notes |
|-----------|-------|-----|---------|-------|
| HellaSwag | Baseline | Regresses | Recovers to near baseline | Commonsense reasoning |
| SQuAD 2 | Baseline | Regresses | Recovers to near baseline | Reading comprehension |
| DROP | Baseline | Regresses | Recovers to near baseline | Discrete reasoning |
| WMT FR→EN | Baseline | Regresses | Recovers to near baseline | Translation |
| Overall NLP | Baseline | Significant drops | Near-baseline performance | Across multiple benchmarks |

> **Key insight:** PPO-ptx shows you can have alignment *without* paying a large alignment tax. The pretraining mix term acts as a regularizer, keeping the model competent at general language tasks while still benefiting from RLHF.

---

## Key Results
*Paper reference: Section 4 — Results (pp. 10–16, Figures 1, 3–7)*

| Metric | 1.3B InstructGPT | 175B GPT-3 | 175B InstructGPT |
|--------|-------------------|------------|------------------|
| Preference rate vs 175B GPT-3 | Preferred (despite 100x fewer params) | Baseline | 85 ± 3% preferred |
| Preference rate vs 175B SFT | — | — | 71 ± 4% preferred over few-shot 175B GPT-3 |
| TruthfulQA (truthful + informative) | — | Baseline | ~2x more truthful + informative |
| Hallucination rate (closed-domain) | — | 41% | 21% |
| Toxicity reduction (respectful prompt) | — | Baseline | ~25% fewer toxic outputs |
| Follows explicit constraints | — | Often ignores | Substantially better adherence |

**Key takeaways:**
- 1.3B PPO-ptx is preferred over 175B GPT-3 — alignment beats scale
- 175B InstructGPT is preferred 85 ± 3% of the time vs 175B GPT-3 in head-to-head comparisons
- 175B InstructGPT is preferred 71 ± 4% vs few-shot 175B GPT-3 (even when GPT-3 gets examples)
- TruthfulQA: PPO models generate approximately twice as many truthful and informative answers as GPT-3
- Hallucination: 21% vs 41% on closed-domain summarization tasks (nearly cut in half)
- Toxicity: ~25% reduction in toxic outputs when given a respectful prompt
- Held-out labelers (not involved in training) show similar preferences, suggesting the model generalizes beyond the specific training labelers

---

## Evaluation Framework
*Paper reference: Section 3.6 — Evaluation (pp. 9–10, Table 3)*

**Helpfulness:** The model should help the user solve their task, follow instructions, and infer intent even when instructions are ambiguous.

**Truthfulness:** The model's statements about the world should be factually true. Measured via TruthfulQA (adversarial questions designed to elicit false answers) and hallucination rate on closed-domain summarization tasks.

**Harmlessness:** The model should not be toxic, biased, or produce harmful content. Measured via RealToxicityPrompts, Winogender (gender bias in pronoun resolution), and CrowS-Pairs (stereotypical bias).

**The honest → truthful distinction:** We cannot measure the "beliefs" of a language model, so the paper measures whether model outputs are factually true (truthfulness) rather than whether the model "believes" what it says (honesty).

> **Key insight:** During training, helpfulness is prioritized — labelers rate responses primarily on whether they follow instructions well. During final evaluations, truthfulness and harmlessness are also measured. These criteria can conflict: a user may ask for harmful content, and a helpful model that complies would fail the harmlessness criterion.

---

## "Who Are We Aligning To?"
*Paper reference: Section 5.2 (pp. 18–19) and Section 5.3 — Limitations (p. 19)*

The human feedback in RLHF comes from a specific group of people:

- ~40 contractors hired from Upwork and ScaleAI
- Mostly English-speaking, living in the US or Southeast Asia
- Inter-annotator agreement: ~73%

**Alignment is shaped by multiple layers of bias:**

1. **Labelers' own values and demographics** — their personal beliefs about what is "good" or "harmful"
2. **Researcher-written instructions** — OpenAI's guidelines shape what labelers reward
3. **The context of being a paid job** — labelers may optimize for speed or consistency rather than deep reflection
4. **API customer prompts** — the training data comes from API users, implicitly aligning the model to customer preferences

**Who is missing:**
- OpenAI's customers are not representative of all users
- The API waitlist was seeded from OpenAI's own networks
- Non-English speakers, non-Western perspectives, and marginalized communities are underrepresented

> **Limitation:** "We are not claiming that researchers, the labelers we hired, or our API customers are the right source of preferences."

---

## Datasets & Benchmarks
*Paper reference: Section 4.2 — Results on public NLP datasets (pp. 13–15, Figures 6–7)*

| Benchmark | What It Measures | Example | InstructGPT Result |
|-----------|-----------------|---------|-------------------|
| **TruthfulQA** | Truthfulness on adversarial questions designed to elicit false answers | "What happens if you crack a mirror?" — GPT-3: "7 years bad luck" | ~2x more truthful + informative than GPT-3 |
| **RealToxicityPrompts** | Toxicity of model continuations given potentially toxic prompts | Model is given a half-sentence and must continue it without producing toxic language | ~25% less toxic with respectful prompt |
| **Winogender** | Gender bias in pronoun resolution with gendered occupations | "The nurse notified the patient that [his/her] shift would be ending" — tests whether model associates occupations with gender | No significant improvement over GPT-3 |
| **CrowS-Pairs** | Stereotypical bias via pairs of sentences differing only in demographic group | Measures whether the model assigns higher probability to the stereotypical version of a sentence | No significant improvement over GPT-3 |

> **Note:** InstructGPT significantly improves on truthfulness and toxicity but does not substantially reduce bias as measured by Winogender and CrowS-Pairs. RLHF aligns to labeler preferences, and labelers were not specifically trained to address subtle stereotypical biases.

---

## Glossary

| Term | Definition |
|------|-----------|
| **RLHF** | Reinforcement Learning from Human Feedback; a technique that uses human preference rankings to train a reward model, which then guides RL optimization of a language model. |
| **PPO** | Proximal Policy Optimization; an RL algorithm that updates the policy in small, stable steps by clipping the objective to prevent large parameter changes. |
| **KL Divergence** | Kullback-Leibler divergence; measures how much one probability distribution differs from another. Used here to penalize the RL policy for drifting from the SFT model. |
| **Reward Model** | A neural network trained on human comparison data to output a scalar score representing how "good" a response is. Replaces the human in the RL loop. |
| **Alignment Tax** | The performance regression on standard NLP benchmarks caused by RLHF fine-tuning. The model gets better at following instructions but worse at general language tasks. |
| **SFT (Supervised Fine-Tuning)** | Fine-tuning a pre-trained model on labeler-written demonstrations of ideal behavior. The first stage of the RLHF pipeline. |
| **Policy (RL context)** | The model's strategy for generating responses. In RLHF, the language model *is* the policy — it maps prompts (states) to responses (actions). |
| **Bandit Environment** | A simplified RL setting with a single step: the agent takes one action (generates a full response) and receives a reward. No sequential decision-making across multiple states. |
| **PPO-ptx** | A variant of PPO that mixes pretraining language modeling gradients into RL training to mitigate the alignment tax. The "ptx" stands for pretraining mix. |
| **Alignment** | The problem of making AI systems behave in accordance with human intentions and values. InstructGPT operationalizes this as matching labeler preferences for helpful, truthful, and harmless outputs. |
