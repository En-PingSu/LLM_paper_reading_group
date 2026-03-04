# Week 1 — Paper Notes
**Paper:** Language Models are Few-Shot Learners (GPT-3)

---

## Overview

This paper is somewhat obsolete by 2026 standards, but it makes you appreciate the magic of prompt writing, which these days we take for granted. It covers many fundamental terms and systems that will help us understand future papers more quickly.

The gist is simple — as the title says, large language models are few-shot learners. If you give multiple examples in the prompt, you get better output via in-context learning. The model "learns"/understands what to produce more accurately. This effect scales with model size. It's a valuable finding because it reduces a lot of the overhead that comes with fine-tuning on each specific task and the resulting lack of generalization. The paper reads more like a discovery of this fact than any concrete explanation of *why* it works.

---

## Things That Came Up During Reading

> *(Add specific observations, confusions, and aha moments here as you read.)*

---

## Key Points

- Few-shot prompting (in-context learning) improves output quality without weight updates
- Performance scales with model size
- Reduces the need for task-specific fine-tuning

---

## Batch Size & Training

Prior work showed that larger neural networks can effectively use larger batch sizes because their gradient estimates become more stable as model size increases, but they also require smaller learning rates to maintain training stability.

To determine an efficient batch size, the authors measure the **gradient noise scale** during training, which compares how much mini-batch gradients vary relative to their average direction. Practically, they compute gradients on several different mini-batches, calculate the mean gradient, measure how much individual gradients deviate from that mean, and use this ratio to estimate a "critical" batch size. They then choose a batch size near this value — increasing it beyond that point gives little additional improvement while wasting computational resources.

---

## Glossary

**FLOPS**
A floating point operation (FLOP) is a single mathematical calculation.

**PetaFLOP/s-days**
A unit that measures total computational work during training. One petaflop equals 10¹⁵ operations per second. To compute petaflop/s-days, multiply the system's compute speed (in petaflops/s) by the number of training days.

*Example:* A cluster running at 10 petaflops/s for 5 days = 10 × 5 = **50 petaflop/s-days**

This metric lets researchers compare overall training effort across different models, regardless of how many GPUs were used, by capturing both computational power and training duration in a single number.

**Scaling Laws**
*(To fill in — see Chinchilla paper)*

**NMT** — Neural Machine Translation

---

## Metrics

### Perplexity

Measures how well a language model predicts the next word in a sequence. Lower perplexity = better predictions. In an autoregressive model, each word is predicted based on the previous words.

**Formula:**
```
Perplexity = exp(−(1/N) Σ log P(wᵢ | w<i))
```

**Example** — sentence: "I love NLP."

| Word | Probability | ln(p) |
|------|------------|-------|
| I | 0.5 | −0.693 |
| love \| I | 0.1 | −2.303 |
| NLP \| I love | 0.2 | −1.609 |

Sum = −4.605 → Average = −1.535 → Negate = 1.535 → e^1.535 ≈ **4.64**

This means the model is, on average, as uncertain as choosing among ~4–5 equally likely words at each step. Perplexity evaluates next-word prediction quality and fluency but does not directly measure meaning or correctness.

---

### BLEU (Bilingual Evaluation Understudy)

Measures how similar generated text is to a reference text. Commonly used in machine translation. Works by comparing overlapping word sequences (n-grams) between generated output and a reference.

**Example:**

- Reference: *"The cat is on the mat"*
- Generated: *"The cat is on mat"*

| N-gram | Match | Precision |
|--------|-------|-----------|
| Unigram | 5/5 | 1.00 |
| Bigram | 3/4 | 0.75 |

BLEU combines n-gram precisions (up to 4-grams) and applies a **brevity penalty** when the generated sentence is shorter than the reference.

**Summary:** Perplexity measures how well a model predicts language using probabilities; BLEU measures how closely generated text matches a reference using n-gram overlap.

---

## Benchmarks & Datasets

### Language Modeling & Cloze

| Dataset | Task | Example |
|---------|------|---------|
| **PTB** (Penn Treebank) | Predict next word | "The stock market crashed after the unexpected" → *announcement* |
| **LAMBADA** | Predict final word using long-range context | "Sarah forgot her umbrella. When it started raining, she got completely …" → *wet* |
| **HellaSwag** | Choose most plausible ending (adversarial) | "A man is assembling a bookshelf…" → *He carefully tightens the screws.* |
| **StoryCloze** | Pick correct final sentence of a 5-sentence story | Tests narrative coherence |

---

### Closed-Book Question Answering

| Dataset | Description | Example |
|---------|-------------|---------|
| **Natural Questions (NQ)** | Real Google search questions answered from Wikipedia | "Who wrote Hamlet?" → *William Shakespeare* |
| **WebQuestions** | Short factual questions from web queries | "What is the capital of Canada?" → *Ottawa* |
| **TriviaQA** | Large trivia dataset; tests broad world knowledge | "Which planet is known as the Red Planet?" → *Mars* |

---

### Translation (WMT)

| Pair | Example |
|------|---------|
| EN → FR | "I love learning." → *J'aime apprendre.* |
| EN → DE | "The book is on the table." → *Das Buch liegt auf dem Tisch.* |
| EN → RO | "Good morning." → *Bună dimineața.* |

---

### Winograd-Style Reasoning

| Dataset | Description | Example |
|---------|-------------|---------|
| **WSC273** | Pronoun resolution requiring commonsense reasoning | "The trophy doesn't fit in the suitcase because **it** is too big." → *The trophy* |
| **Winogrande** | Harder, adversarial version | "Alex helped Jordan because **he** was kind." → *Alex* |

---

### Commonsense & Physical Reasoning

| Dataset | Description | Example |
|---------|-------------|---------|
| **PIQA** | Everyday physical interactions | "How do you dry wet clothes?" → *Put them in a dryer* |
| **ARC** | Science exam questions | "What gas do plants absorb?" → *Carbon dioxide* |
| **OpenBookQA** | Science reasoning requiring known facts | Fact: Metals conduct electricity → "Best wire material?" → *Copper* |

---

### Reading Comprehension

| Dataset | Description | Example |
|---------|-------------|---------|
| **CoQA** | Conversational QA over a passage | Passage: "Anna moved to Paris in 2010." → "When did Anna move?" → *2010* |
| **DROP** | Reading comprehension + arithmetic reasoning | "John had 5 apples and bought 3 more." → *8* |
| **QuAC** | Conversational reading comprehension | Passage about Lincoln → "When was he born?" → *1809* |
| **SQuAD v2** | Extractive QA with unanswerable questions | "Where is the Statue of Liberty?" (from a Paris passage) → *No answer* |
| **RACE** | Multi-paragraph reading comprehension exams | Passage-based multiple choice |

---

### GLUE Benchmark
*General Language Understanding Evaluation — 9 tasks*

| Task | Description | Example |
|------|-------------|---------|
| **CoLA** | Grammatical acceptability | "The cat are sleeping." → *Ungrammatical* |
| **SST-2** | Sentiment analysis | "The movie was fantastic!" → *Positive* |
| **MRPC** | Paraphrase detection | "The company released earnings." ≈ "The firm published results." → *Paraphrase* |
| **STS-B** | Semantic similarity score | "A man is playing guitar." vs "A person is playing music." → *High similarity* |
| **QQP** | Duplicate question detection | "How to lose weight?" vs "How can I reduce body fat?" → *Duplicate* |
| **MNLI** | Multi-genre natural language inference | "Dogs are animals." / "Dogs are mammals." → *Entailment* |
| **QNLI** | Question NLI | Passage + question entailment |
| **RTE** | Recognizing textual entailment | Small entailment dataset |
| **WNLI** | Winograd NLI | Pronoun reasoning version |

---

### SuperGLUE Benchmark
*Harder successor to GLUE, designed after models nearly solved GLUE*

| Task | Description | Example |
|------|-------------|---------|
| **BoolQ** | Yes/No QA | "Can penguins fly?" (Passage: "Penguins cannot fly.") → *No* |
| **COPA** | Cause/Effect choice | "The ground was wet because…" → *It rained* |
| **MultiRC** | Multi-sentence reasoning | Multiple correct answers allowed |
| **ReCoRD** | Reading comprehension with entities | Fill missing entity in passage |
| **RTE** | Textual entailment | Premise–hypothesis inference |
| **WiC** | Word in Context | "Bank" (river vs money) → *Different meaning* |
| **WSC** | Winograd Schema | Pronoun resolution reasoning |

---

### NLI — Natural Language Inference

| Dataset | Description | Example |
|---------|-------------|---------|
| **ANLI** | Adversarial NLI; tests logical reasoning | Premise: "All birds can fly." / Hypothesis: "Penguins can fly." → *Contradiction* |

---

## Notes on Tokenizers

Modern language models do not learn their tokenizers with gradient descent. Tokenization is a separate preprocessing step built using frequency-based algorithms **before** neural network training begins.

**Byte Pair Encoding (BPE)** — used in GPT-2 and RoBERTa
Starts from characters and repeatedly merges the most frequent adjacent pair.

*Example:* `l o w e r` → (merge "lo") → `lo w e r` → continues until a fixed vocabulary size is reached.

**Other popular tokenizers:**
- **WordPiece** — used in BERT
- **SentencePiece (Unigram LM)** — used in T5 and LLaMA

A single tokenizer often does not transfer well across languages because its merge rules are learned from language-specific frequency statistics. Differences in scripts, morphology, and character distributions (e.g., English vs. Chinese or Arabic) can lead to inefficient segmentations and longer token sequences unless the tokenizer is trained on multilingual data.
