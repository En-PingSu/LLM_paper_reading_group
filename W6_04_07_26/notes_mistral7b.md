# Week 6 (Paper 1) — Paper Notes
**Paper:** Mistral 7B, Jiang et al. 2023 (Mistral AI)

---

## Table of Contents

1. [Overview](#overview)
2. [Things That Came Up During Reading](#things-that-came-up-during-reading)
3. [Key Points](#key-points)
4. [Architecture](#architecture)
   - [Model Parameters](#model-parameters)
   - [Sliding Window Attention (SWA)](#sliding-window-attention-swa)
   - [Rolling Buffer Cache](#rolling-buffer-cache)
   - [Pre-fill and Chunking](#pre-fill-and-chunking)
   - [Grouped-Query Attention (GQA)](#grouped-query-attention-gqa)
5. [Benchmark Results](#benchmark-results)
   - [Performance vs. LLaMA 2 and Code Llama](#performance-vs-llama-2-and-code-llama)
   - [Size and Efficiency Analysis](#size-and-efficiency-analysis)
6. [Instruction Finetuning](#instruction-finetuning)
7. [Guardrails and Safety](#guardrails-and-safety)
   - [System Prompt Engineering](#system-prompt-engineering)
   - [Content Moderation with Self-Reflection](#content-moderation-with-self-reflection)
8. [Connections to Previous Weeks](#connections-to-previous-weeks)
9. [Glossary](#glossary)

---

## Overview
*Paper reference: Abstract & Section 1 (pp. 1–2)*

Mistral 7B is a 7.3 billion parameter language model released by Mistral AI that outperforms all existing open-source models up to 13B parameters on every benchmark evaluated, and even approaches the performance of models 3-5x its size. The paper's central claim is provocative: **language models may compress knowledge more than what was previously thought**, meaning that smaller models can encode far more capability per parameter than the field had assumed.

The paper introduces two architectural innovations on top of the standard transformer design: **Sliding Window Attention (SWA)**, which replaces full quadratic attention with a fixed-window local attention that still achieves long-range information flow through layer stacking, and **Grouped-Query Attention (GQA)**, which reduces the KV cache memory footprint by sharing key-value heads across groups of query heads. Together with a **Rolling Buffer Cache**, these mechanisms make Mistral 7B both faster and more memory-efficient at inference time compared to standard transformer models of similar size.

The model is released under the **Apache 2.0 license** with no restrictions on use, making it one of the most permissively licensed high-performance open models at its time of release. The authors also release Mistral 7B Instruct, a chat-fine-tuned variant that outperforms all 7B and 13B chat models available at the time of publication.

---

## Things That Came Up During Reading

> *(Add specific observations, confusions, and aha moments here as you read.)*

---

## Key Points
*Paper reference: Sections 1–5 (pp. 1–6)*

- Mistral 7B (7.3B params) **outperforms LLaMA 2 13B on all benchmarks** and matches or beats LLaMA 2 34B on many tasks — demonstrating that architecture and training quality can compensate for raw parameter count
- Uses **Sliding Window Attention** with window size $W = 4096$ to reduce attention from $O(n^2)$ to $O(n \times W)$, while achieving a theoretical attention span of ~131K tokens through layer stacking
- **Rolling Buffer Cache** fixes the KV cache size at $W$ entries regardless of sequence length, reducing cache memory by up to 8x for long sequences
- Employs **Grouped-Query Attention** with 8 KV heads shared across 32 query heads (4:1 ratio), reducing memory bandwidth during decoding
- On MMLU, Mistral 7B scores **60.1%** vs. LLaMA 2 7B's 44.4% and LLaMA 2 13B's 55.6% — a 15.7-point improvement over the same-size competitor
- Achieves **52.2% on GSM8K** (math reasoning), tripling LLaMA 2 7B's 16.0% and exceeding LLaMA 2 13B's 34.3%
- On code tasks, scores **30.5% HumanEval** and **47.5% MBPP**, approaching the specialized Code-Llama 7B (31.1% / 52.5%) despite being a general-purpose model
- The "effective LLaMA 2 size" analysis shows Mistral 7B is equivalent to a ~23B model on MMLU, ~38B on reasoning, and ~13B on knowledge — a **3.3x compression ratio** on MMLU (up to 5.4x on reasoning)
- Mistral 7B Instruct outperforms Llama 2 13B Chat on both **MT-Bench** (6.84 vs. 6.65) and **Chatbot Arena ELO** (1031 vs. 1012) using only publicly available fine-tuning data
- Demonstrates a novel **self-reflection guardrail** where the model classifies its own outputs for safety, achieving 99.4% precision and 95.6% recall
- The paper is notably concise (9 pages, with only 6 pages of main content) and reveals **no details about training data, training procedure, or compute** — a sharp contrast to LLaMA's transparency
- Released under **Apache 2.0** — the most permissive license among high-performance open models at the time

---

## Architecture
*Paper reference: Section 2 (pp. 2–3)*

### Model Parameters

Mistral 7B is based on the same transformer backbone as LLaMA (RMSNorm pre-normalization, SwiGLU activation, Rotary Positional Embeddings) with two key architectural additions: Sliding Window Attention and Grouped-Query Attention.

| Parameter | Value |
|-----------|-------|
| Dimension ($d_{\mathrm{model}}$) | 4,096 |
| Number of layers ($n_{\mathrm{layers}}$) | 32 |
| Head dimension ($d_{\mathrm{head}}$) | 128 |
| Hidden dimension ($d_{\mathrm{ff}}$) | 14,336 |
| Number of attention heads ($n_{\mathrm{heads}}$) | 32 |
| Number of KV heads ($n_{\mathrm{kv\_heads}}$) | 8 |
| Sliding window size ($W$) | 4,096 |
| Context length | 8,192 |
| Vocabulary size | 32,000 |

> **Comparison to LLaMA (W5):** Mistral 7B uses the same $d_{\mathrm{model}} = 4096$, 32 layers, and 32,000 vocab as LLaMA 1 7B and Llama 2 7B. The key differences are: (1) GQA with 8 KV heads instead of 32 (LLaMA 2 only used GQA for 34B+), (2) Sliding Window Attention with $W = 4096$, (3) a larger FFN hidden dim of 14,336 (vs. 11,008 in LLaMA 7B), and (4) 8,192 context length (vs. 4,096 in Llama 2 7B). The larger FFN dimension partly explains the higher parameter count (7.3B vs. 6.7B).

---

### Sliding Window Attention (SWA)

Standard (vanilla) self-attention computes attention between every pair of tokens in a sequence, resulting in $O(n^2)$ time and memory complexity where $n$ is the sequence length. For long sequences, this becomes prohibitively expensive.

**SWA restricts each token to attend only to the previous $W$ tokens** (where $W = 4096$ is the window size), reducing the per-layer cost to $O(n \times W)$. However, because information propagates through layers, the **effective attention span grows linearly with depth**.

#### How Information Propagates

At layer 1, token $t$ can attend to tokens in the range $[t - W, t]$. At layer 2, token $t$ can attend to tokens that themselves attended to tokens at range $[t - 2W, t]$ at layer 1. After $k$ layers:

$$\mathrm{effective\_span}(k) = k \times W$$

Where:
- $k$ = number of transformer layers the information has passed through
- $W$ = sliding window size (4,096 for Mistral 7B)

For Mistral 7B with 32 layers and $W = 4096$:

$$\mathrm{effective\_span}(32) = 32 \times 4096 = 131{,}072 \text{ tokens}$$

This means that while each individual attention layer is local, the model can in principle propagate information across **~131K tokens** through the full depth of the network — far beyond the 8,192-token context window.

![Figure 1: Sliding Window Attention](figures/mistral7b_fig1_sliding_window_attention.png)

*Figure 1: Comparison of attention patterns. Left: Vanilla attention where every token attends to all previous tokens (O(n^2)). Center: Sliding window attention where each token attends to W previous tokens. Right: The effective attention span after stacking layers — information from token 1 can reach token 3W+1 after 3 layers.*

```
Vanilla Attention (layer k):          Sliding Window Attention (layer k):
Token 8 attends to: [1,2,3,4,5,6,7]  Token 8 attends to: [5,6,7] only (W=3)

But after 3 SWA layers:
Layer 1: token 8 sees [5,6,7,8]
Layer 2: token 8 sees [5,6,7,8] → those saw [2,3,4,5,6,7,8] at layer 1
Layer 3: effective range reaches back to token 1 if 3 × W ≥ 8
```

**Speed benefit:** For a sequence length of 16,384 tokens with $W = 4096$, using FlashAttention and xFormers with SWA yields a **2x speed improvement** over vanilla attention.

---

### Rolling Buffer Cache

During autoregressive generation, the standard approach caches all past key-value pairs, growing linearly with sequence length. Mistral 7B instead uses a **fixed-size rolling buffer** of size $W$.

Keys and values for position $i$ are stored at cache index:

$$\mathrm{cache\_index}(i) = i \mod W$$

Where:
- $i$ = the absolute position of the token in the sequence
- $W$ = the window size (4,096)

When $i > W$, new entries overwrite the oldest ones — but those old entries are outside the attention window anyway, so the model never needs them.

![Figure 2: Rolling Buffer Cache](figures/mistral7b_fig2_rolling_buffer_cache.png)

*Figure 2: The rolling buffer cache. Keys and values are stored at position i mod W. When position i exceeds W, old values are overwritten. This keeps the cache at a constant size of W regardless of sequence length.*

**Worked example:**

For $W = 4$ (simplified):

| Token position $i$ | Cache index ($i \mod 4$) | Overwrites position |
|---------------------|--------------------------|---------------------|
| 0 | 0 | — |
| 1 | 1 | — |
| 2 | 2 | — |
| 3 | 3 | — |
| 4 | 0 | position 0 |
| 5 | 1 | position 1 |
| 6 | 2 | position 2 |

At position 6, the cache contains entries for positions {3, 4, 5, 6} — exactly the $W = 4$ most recent tokens.

**Memory savings:** For a sequence of length 32,768 with $W = 4096$, the rolling buffer cache is **8x smaller** than a full attention cache. This has no impact on model quality because tokens outside the window are never attended to.

---

### Pre-fill and Chunking

During generation, the model first processes the entire prompt (the "pre-fill" phase) before generating tokens autoregressively. For long prompts, Mistral 7B **chunks the prompt into pieces of size $W$** and processes each chunk sequentially.

Each chunk attends to:
1. The **cached KV pairs** from the rolling buffer (tokens from previous chunks that are still within the window)
2. **Itself** with a causal attention mask

This means the attention mask for each chunk has three regions:

| Region | Description | Attention values |
|--------|-------------|-----------------|
| **Past (out of window)** | Tokens that have been evicted from the rolling buffer | Zeros (no attention) |
| **Cache (in window)** | Previously processed tokens still in the buffer | Sliding window attention |
| **Current chunk** | Tokens in the current chunk | Causal (lower-triangular) |

This chunked approach means the prompt pre-fill never requires more than $O(W^2)$ memory for attention, regardless of total prompt length.

---

### Grouped-Query Attention (GQA)

Standard Multi-Head Attention (MHA) allocates separate key and value projections for each query head. Mistral 7B uses **Grouped-Query Attention**, where multiple query heads share a single key-value head:

- **32 query heads** but only **8 KV heads**
- Every group of **4 query heads** shares the same KV head
- KV cache size is reduced by a factor of $32 / 8 = 4$

```
Standard Multi-Head Attention (32 heads):
Q₁→KV₁  Q₂→KV₂  Q₃→KV₃  ... Q₃₂→KV₃₂     (32 KV heads)

Grouped-Query Attention (Mistral 7B):
Q₁ ─┐         Q₅ ─┐         Q₉ ─┐              Q₂₉─┐
Q₂ ─┤→ KV₁   Q₆ ─┤→ KV₂   Q₁₀─┤→ KV₃   ...  Q₃₀─┤→ KV₈
Q₃ ─┤         Q₇ ─┤         Q₁₁─┤              Q₃₁─┤
Q₄ ─┘         Q₈ ─┘         Q₁₂─┘              Q₃₂─┘
              (8 KV heads, 4 Q heads per group)
```

The **KV cache memory** per token per layer is:

$$\mathrm{KV\_cache} = 2 \times n_{\mathrm{kv\_heads}} \times d_{\mathrm{head}} = 2 \times 8 \times 128 = 2{,}048 \text{ values}$$

Where:
- $2$ = one for keys, one for values
- $n_{\mathrm{kv\_heads}} = 8$ = number of key-value heads
- $d_{\mathrm{head}} = 128$ = dimension per head

Compared to full MHA ($n_{\mathrm{kv\_heads}} = 32$), this is a **4x reduction** in KV cache size, which directly enables higher batch sizes and throughput during inference.

> **Comparison to Llama 2 (W5):** Llama 2 only used GQA for its 34B and 70B models — the 7B and 13B models used standard MHA. Mistral 7B applies GQA even at the 7B scale, showing that the quality tradeoff is worthwhile even for smaller models when combined with the memory savings from Sliding Window Attention.

---

## Benchmark Results
*Paper reference: Section 3 (pp. 3–5)*

### Benchmark Descriptions

| Benchmark | What It Tests | Format | Metric | Shot Setting |
|-----------|--------------|--------|--------|-------------|
| **MMLU** | Multitask language understanding (57 subjects) | Multiple choice | Accuracy (%) | 5-shot |
| **HellaSwag** | Commonsense reasoning (sentence completion) | Multiple choice | Accuracy (%) | 0-shot |
| **WinoGrande** | Commonsense reasoning (pronoun resolution) | Binary choice | Accuracy (%) | 0-shot |
| **PIQA** | Physical intuition (physical commonsense) | Binary choice | Accuracy (%) | 0-shot |
| **ARC-Easy** | Grade-school science questions (easy set) | Multiple choice | Accuracy (%) | 0-shot |
| **ARC-Challenge** | Grade-school science questions (hard set) | Multiple choice | Accuracy (%) | 0-shot |
| **NaturalQuestions** | Open-domain question answering (Wikipedia) | Open-ended | Exact match (%) | 5-shot |
| **TriviaQA** | Trivia knowledge question answering | Open-ended | Exact match (%) | 5-shot |
| **HumanEval** | Code generation (Python functions from docstrings) | Code generation | pass@1 (%) | 0-shot |
| **MBPP** | Code generation (Python from descriptions) | Code generation | pass@1 (%) | 3-shot |
| **MATH** | Competition-level mathematics | Open-ended | Accuracy (%, maj@4) | 4-shot |
| **GSM8K** | Grade-school math word problems | Open-ended | Accuracy (%, maj@8) | 8-shot |

### Performance vs. LLaMA 2 and Code Llama

![Figure 3: Benchmark comparisons across models](figures/mistral7b_fig3_benchmark_comparison.png)

*Figure 3: Bar chart comparisons of Mistral 7B against LLaMA 2 7B, LLaMA 2 13B, and Code Llama 7B across multiple benchmark categories.*

| Model | Modality | MMLU | HellaSwag | WinoG. | PIQA | ARC-e | ARC-c | NQ | TriviaQA | HumanEval | MBPP | MATH | GSM8K |
|-------|----------|------|-----------|--------|------|-------|-------|----|----------|-----------|------|------|-------|
| LLaMA 2 7B | Pretrained | 44.4 | 77.1 | 69.5 | 77.9 | 68.7 | 43.2 | 24.7 | 63.8 | 11.6 | 26.1 | 3.9 | 16.0 |
| LLaMA 2 13B | Pretrained | 55.6 | 80.7 | 72.9 | 80.8 | 75.2 | 48.8 | 29.0 | 69.6 | 18.9 | 35.4 | 6.0 | 34.3 |
| Code Llama 7B | Finetuned | 36.9 | 62.9 | 62.3 | 72.8 | 59.4 | 34.5 | 11.0 | 34.9 | 31.1 | 52.5 | 5.2 | 20.8 |
| **Mistral 7B** | **Pretrained** | **60.1** | **81.3** | **75.3** | **83.0** | **80.0** | **55.5** | **28.8** | **69.9** | **30.5** | **47.5** | **13.1** | **52.2** |

**Key takeaways:**

- **Mistral 7B beats LLaMA 2 13B on every single benchmark** despite having roughly half the parameters (7.3B vs. 13B)
- The MMLU gap is striking: 60.1% vs. 44.4% (LLaMA 2 7B) — a **+15.7 point** improvement at the same scale
- GSM8K (math reasoning) shows a **3.3x improvement** over LLaMA 2 7B: 52.2% vs. 16.0%
- On code benchmarks, Mistral 7B nearly matches the **specialized** Code Llama 7B (30.5% vs. 31.1% on HumanEval) while dramatically outperforming it on all non-code tasks
- The commonsense reasoning cluster (HellaSwag, WinoGrande, PIQA, ARC) shows consistent 3-12 point improvements over LLaMA 2 13B

### Size and Efficiency Analysis

The paper presents a compelling "equivalent model size" analysis, interpolating from LLaMA 2's performance scaling curve to estimate what size LLaMA 2 model would be needed to match Mistral 7B.

![Figure 4: Performance scaling — effective LLaMA 2 size](figures/mistral7b_fig4_performance_scaling.png)

*Figure 4: Estimated "effective LLaMA 2 size" of Mistral 7B across different capability categories. Mistral 7B with 7.3B parameters performs like much larger LLaMA 2 models depending on the task.*

| Capability | Mistral 7B equivalent LLaMA 2 size | Compression ratio |
|-----------|-------------------------------------|-------------------|
| **MMLU** (Knowledge + Reasoning) | ~23B | 3.3x |
| **Commonsense Reasoning** | ~38B | 5.4x |
| **World Knowledge** | ~13B | 1.9x |
| **Reading Comprehension** | ~21B | 3.0x |

The compression ratios are highest for **reasoning tasks** (5.4x) and lowest for **world knowledge** (1.9x). This makes intuitive sense: reasoning ability is more about learned computation patterns (which can be compressed), while world knowledge is more about memorizing facts (which scales more directly with parameter count).

> **Comparison to LLaMA (W5):** LLaMA 1's key insight was that smaller models trained on more data could match larger models trained on less data (the Chinchilla-optimal regime). Mistral 7B pushes this further — not just through training data quantity, but through **architectural efficiency** (SWA, GQA) and presumably better training data quality (though the paper reveals nothing about the training data).

---

## Instruction Finetuning
*Paper reference: Section 4 (pp. 4–5)*

Mistral 7B Instruct is a chat-optimized version fine-tuned on **publicly available instruction datasets from HuggingFace**. The authors emphasize that no proprietary data or special training tricks were used — this is a baseline to demonstrate that the pretrained model is a strong foundation for fine-tuning.

### Benchmark Descriptions

| Benchmark | What It Tests | Format | Metric |
|-----------|--------------|--------|--------|
| **MT-Bench** | Multi-turn conversation quality across 8 categories | Open-ended dialogue (2 turns) | GPT-4 judge score (1–10) |
| **Chatbot Arena ELO** | Crowdsourced head-to-head model comparisons | Side-by-side blind evaluation | ELO rating |

### Instruct Model Comparison

| Model | Size | Chatbot Arena ELO | MT-Bench |
|-------|------|-------------------|----------|
| WizardLM 13B v1.2 | 13B | 1047 | 7.20 |
| **Mistral 7B Instruct** | **7B** | **1031** | **6.84 $\pm$ 0.07** |
| Vicuna 13B | 13B | 1041 | 6.57 |
| Llama 2 13B Chat | 13B | 1012 | 6.65 |
| Vicuna 7B | 7B | 997 | 6.17 |
| Llama 2 7B Chat | 7B | 985 | 6.27 |
| Alpaca 13B | 13B | 914 | 4.53 |

**Key observations:**

- Mistral 7B Instruct **outperforms all 13B chat models** except WizardLM 13B v1.2 on ELO, and beats every 13B model except WizardLM on MT-Bench
- It surpasses Llama 2 13B Chat by **+19 ELO points** and **+0.19 on MT-Bench** despite being nearly half the size
- Compared to Llama 2 7B Chat, the improvement is **+46 ELO** and **+0.57 MT-Bench**
- In a **human evaluation** against Llama 2 13B Chat, Mistral 7B Instruct was preferred **5,020 times vs. 4,143** (55% vs. 45%), showing a meaningful human preference even beyond automated metrics

> **Comparison to InstructGPT (W4):** InstructGPT showed that SFT + RLHF could make a 1.3B model competitive with a 175B base model on human preference. Mistral 7B Instruct achieves strong results with just SFT on public data — no RLHF at all — suggesting that a sufficiently strong pretrained model may need less alignment effort. However, the authors note this is just a baseline and more sophisticated fine-tuning would likely improve results further.

---

## Guardrails and Safety
*Paper reference: Section 5 (pp. 5–6)*

### System Prompt Engineering

The paper explores using system prompts to enforce safe behavior without degrading helpfulness. Three configurations are tested:

| Configuration | System prompt | MT-Bench score |
|--------------|---------------|----------------|
| No system prompt | (none) | 6.84 |
| Llama 2 system prompt | Long, detailed safety instructions | 6.38 |
| Mistral recommended prompt | "Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity." | 6.58 |

**Key finding:** With the recommended system prompt, Mistral 7B Instruct achieves **100% refusal rate** on a set of 175 unsafe prompts. The Mistral prompt causes less MT-Bench degradation than the Llama 2 prompt (6.58 vs. 6.38, compared to 6.84 with no prompt), demonstrating that a concise prompt can be more effective than a verbose one.

The tradeoff is visible: any safety prompt reduces helpfulness on benign tasks (6.84 → 6.58), but the Mistral prompt minimizes this cost while still achieving full unsafe content refusal.

### Content Moderation with Self-Reflection

The paper proposes using Mistral 7B Instruct itself as a **content moderator** through prompt engineering — a technique the authors call "self-reflection."

The model is prompted to classify whether a given user prompt or model response is acceptable or falls into one of several harmful categories:

| Category | Description |
|----------|-------------|
| **Illegal activities** | Terrorism, child abuse, content illegal under the user's jurisdiction |
| **Hateful content** | Discrimination, slurs, promotion of hatred based on identity |
| **Self-harm** | Promotion or instruction for self-harm or suicide |
| **Unqualified advice** | Medical, legal, financial, or safety advice without proper disclaimers |

The model is given a prompt like:

> "Does this conversation contain any of the above harmful content? Answer with 'yes' or 'no' and explain."

**Self-reflection performance:**

| Metric | Value |
|--------|-------|
| **Precision** | 99.4% |
| **Recall** | 95.6% |

This means the model almost never incorrectly flags safe content as harmful (0.6% false positive rate), while catching 95.6% of actually harmful content. This is a practical and cost-effective alternative to training a separate safety classifier — the same model can both generate and moderate content.

> **Comparison to Llama 2 (W5):** Llama 2-Chat invested heavily in safety: 350+ red teamers, separate safety reward models, safety-specific RLHF, and context distillation. Mistral 7B takes a much lighter approach — system prompts and self-reflection — achieving strong safety results without any safety-specific training. This suggests that much of the safety behavior can be extracted at inference time through careful prompting, at least for straightforward refusal cases.

---

## Connections to Previous Weeks

> **Attention Is All You Need (W2):** Mistral 7B's Sliding Window Attention is a direct modification of the original scaled dot-product attention from the Transformer paper. The original attention computed $\mathrm{softmax}(QK^T / \sqrt{d_k})V$ across all positions. SWA simply masks out entries where the key position is more than $W$ positions behind the query, converting the $O(n^2)$ full attention into $O(n \times W)$ windowed attention. The fundamental mechanism (queries, keys, values, scaled dot product) is unchanged — only the attention mask differs.

> **GPT-1/GPT-2 (W3) and GPT-3 (W1):** The GPT series established the decoder-only, autoregressive language model paradigm that Mistral 7B follows. GPT-3 showed that scaling parameters improves few-shot performance. Mistral 7B challenges the "just scale up" narrative by showing that architectural improvements (SWA, GQA) combined with better training can achieve GPT-3-level benchmark scores with 25x fewer parameters.

> **LLaMA 1 & Llama 2 (W5):** Mistral 7B is most directly comparable to the LLaMA family. It uses the same base modifications (RMSNorm, SwiGLU, RoPE) but adds SWA and uses GQA at the 7B scale (Llama 2 only used GQA at 34B+). LLaMA 1's insight was that more training data can substitute for more parameters. Mistral 7B extends this with the claim that architectural efficiency further compresses knowledge per parameter. The paper is also a stark contrast in transparency: LLaMA papers detailed their training data composition, compute costs, and training procedures, while Mistral 7B reveals essentially nothing about its training setup.

> **InstructGPT / RLHF (W4):** Mistral 7B Instruct demonstrates that SFT alone on public data can produce a competitive chat model — no RLHF required. This contrasts with InstructGPT's multi-stage pipeline (SFT → RM → PPO) and Llama 2-Chat's elaborate iterative RLHF with dual reward models. The self-reflection safety approach is also conceptually related to RLHF's goal of alignment, but achieved through inference-time prompting rather than training-time optimization.

---

## Glossary

| Term | Definition |
|------|-----------|
| **Sliding Window Attention (SWA)** | Attention mechanism where each token attends only to the previous $W$ tokens, reducing complexity from $O(n^2)$ to $O(n \times W)$ |
| **Rolling Buffer Cache** | Fixed-size KV cache that overwrites old entries using modular indexing ($i \mod W$), keeping memory constant regardless of sequence length |
| **Grouped-Query Attention (GQA)** | Attention variant where multiple query heads share a single key-value head, reducing KV cache size. Mistral 7B uses 4 query heads per KV head |
| **FlashAttention** | Memory-efficient exact attention algorithm that avoids materializing the full $n \times n$ attention matrix, instead computing attention in blocks |
| **xFormers** | Meta's library for efficient transformer components, used here in combination with FlashAttention for SWA acceleration |
| **KV Cache** | During autoregressive generation, previously computed key and value vectors are cached to avoid recomputation. Size scales with sequence length in standard attention, but is fixed at $W$ with rolling buffer |
| **Pre-fill** | The initial phase of generation where the entire prompt is processed to populate the KV cache, before token-by-token autoregressive decoding begins |
| **RMSNorm** | Root Mean Square Layer Normalization — a simplified LayerNorm that normalizes by RMS only (no mean centering), used in LLaMA and Mistral architectures |
| **SwiGLU** | Gated Linear Unit activation with Swish gating, replacing standard ReLU/GELU in the FFN. Introduced by Shazeer (2020), adopted by LLaMA and Mistral |
| **RoPE** | Rotary Position Embedding — encodes position information by rotating query and key vectors, enabling relative position awareness. Compatible with SWA |
| **MT-Bench** | Multi-turn benchmark with 80 questions across 8 categories, judged by GPT-4 on a 1-10 scale |
| **Chatbot Arena ELO** | Crowdsourced platform where users compare anonymous model outputs head-to-head, producing ELO rankings |
| **Apache 2.0** | Permissive open-source license allowing commercial use, modification, and distribution with attribution |
| **Self-reflection** | Technique where the model evaluates its own outputs (or inputs) for safety, effectively acting as its own content moderator |
| **maj@k** | Majority voting over $k$ samples — generate $k$ answers and take the most common one. Used for math benchmarks (e.g., maj@8 for GSM8K) |
