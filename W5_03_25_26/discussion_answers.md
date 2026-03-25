# Week 5 — Discussion Questions & Suggested Answers
**Papers:** LLaMA (Touvron et al., Feb 2023) & Llama 2 (Touvron et al., Jul 2023)

These are suggested answers to guide discussion, not definitive answers. Many of these questions are deliberately open-ended.

---

## Scaling & Efficiency (LLaMA 1)

**1. Training budget vs. inference budget — who wins?**

It depends on the use case. If you're training a model for a research paper (run once, evaluated, then discarded), training efficiency dominates — Chinchilla is right. But if you're deploying a model to serve millions of users 24/7, inference efficiency dominates — LLaMA is right. The math is straightforward: a model trained for 3 months but served for 3 years will spend 12x more compute on inference than training. In this regime, paying 2x more to train a model that's 2x cheaper to serve is a massive net win. The industry clearly agreed: after LLaMA, virtually all deployed models have been "over-trained" relative to Chinchilla's recommendations.

**2. Can you train forever?**

Not without limits, but the limits are further than previously thought. There are several constraints:
- **Data exhaustion:** At some point you run out of high-quality data and start repeating. Repeated data has diminishing returns and can cause memorization/overfitting.
- **Model capacity:** A 7B model has finite representational capacity — it can't store or reason about arbitrarily complex knowledge. At some point, more data won't help because the model can't learn from it.
- **Scaling laws predict diminishing returns:** Loss improvement follows a power law — each doubling of data gives less improvement than the last.

In practice, later work (e.g., Llama 3) trained much longer (15T+ tokens) and continued seeing improvements, suggesting the practical limit is primarily data availability, not model capacity for models of this size.

**3. Does LLaMA make Chinchilla wrong?**

No — Chinchilla is answering a different question. Chinchilla asks: "Given a fixed FLOP budget, what's the optimal model-size-to-data ratio?" LLaMA asks: "Given a target inference cost, what model should I train?" These are both valid but different optimization targets. Chinchilla's scaling laws are mathematically correct for what they measure. LLaMA's insight is that the *objective function should change* — optimize for the thing you care about (inference cost) rather than the thing that's convenient to measure (training compute). In practice, the industry has sided with LLaMA's framing, and the Chinchilla-optimal point is now seen as the *minimum* amount of training, not the target.

---

## Open-Source vs. Closed-Source

**4. What does "only public data" actually mean?**

"Publicly available" means the data can be accessed without requiring proprietary agreements — it's a legal/access claim, not an ethical one. CommonCrawl is publicly available, but it contains:
- Copyrighted text (newspapers, books, blog posts) scraped without consent
- Personal information (names, emails, phone numbers)
- Content from communities that didn't agree to be used for AI training

The ethical dimensions are still being litigated (literally — multiple lawsuits are ongoing). However, LLaMA's claim is pragmatically important: anyone can reproduce the training pipeline without needing access to proprietary datasets like those used by GPT-3 or PaLM. This is a significant step toward reproducible research even if the ethical questions remain unresolved.

**5. Is open-sourcing model weights responsible?**

Both sides have merit:
- **For open release:** Security through transparency (the community can find flaws), democratized access (not only big tech can build on frontier models), reproducible research, enables safety research by external groups
- **Against open release:** Bad actors can fine-tune away safety guardrails, weights can't be un-released if problems are found, smaller organizations may deploy without adequate safety testing

The Llama 2 paper argues that openness is net positive for safety because it enables broader scrutiny. In practice, the post-LLaMA ecosystem showed both sides: the open community built impressive safety tools and identified vulnerabilities, but also produced "uncensored" fine-tunes with guardrails removed. The debate remains unresolved.

**6. Did LLaMA change the industry?**

Yes, decisively. Before LLaMA (Feb 2023), no open model was competitive with GPT-3. Within months of LLaMA's leak/release:
- Stanford's Alpaca showed you could instruction-tune LLaMA-7B for ~$600
- Vicuna and others demonstrated competitive chat performance via fine-tuning
- QLoRA showed efficient fine-tuning on consumer hardware
- An ecosystem of local LLM deployment tools (llama.cpp, etc.) emerged

This triggered a "Cambrian explosion" of open-source LLM development that fundamentally changed the industry. Whether this was net positive depends on your threat model, but it unquestionably accelerated both capabilities and accessibility.

---

## Architecture Choices

**7. Why do RMSNorm, SwiGLU, and RoPE matter?**

- **RMSNorm (2019):** Replaces LayerNorm. Simpler (no mean computation, no bias), slightly faster, equal stability. Adoption was slow because LayerNorm worked "well enough" and changing normalization is risky — it can destabilize training. GPT-3 first showed pre-norm was important; RMSNorm simplified it further.
- **SwiGLU (2020):** Replaces ReLU in FFN. Gated activations learn more complex functions. The improvement is empirical — there's no strong theoretical reason why gating helps, but it consistently improves benchmarks by 1-3%. Adoption required changing FFN dimensions ($\frac{2}{3} \times 4d$), which affected many engineering systems.
- **RoPE (2021):** Replaces learned absolute position embeddings. Enables better length generalization and injects position information at every attention layer, not just the input. Particularly important as context lengths grew beyond the original 512-2048 tokens.

These took years to adopt because: (1) each change risks training instability, (2) large-scale experiments are expensive, and (3) the improvements are incremental (1-5%), so the risk/reward seemed low until enough evidence accumulated.

**8. Why use Grouped-Query Attention only for 34B+?**

GQA's primary benefit is reducing the **KV cache size** during inference. For a 7B model, the KV cache is already manageable — each token adds ~0.5MB to the cache. For a 70B model, the KV cache is ~10x larger, making it the dominant memory bottleneck. GQA provides a 4-8x reduction in KV cache size, which is critical for serving 70B models but unnecessary overhead for 7B models where the quality loss (however small) isn't justified by the memory savings.

**9. Is the architecture still the bottleneck?**

Probably not for standard performance. The same decoder-only transformer has been used from GPT-2 (2019) through the most capable models today with only minor modifications. The evidence suggests the bottleneck is primarily **data quality and quantity**, followed by **alignment techniques**, followed by **scale**. Architectural innovations still matter at the margins (FlashAttention, mixture of experts) but the fundamental computation pattern has been stable for 6+ years.

---

## Training Data & Data Curation

**10. CommonCrawl is 67% of LLaMA's diet. Is this a problem?**

It's a tradeoff. CommonCrawl provides enormous scale and diversity but introduces:
- Web-specific biases (SEO spam, marketing language, particular writing styles)
- Factual errors and outdated information
- Toxic and hateful content (partially but not fully filtered)
- English-dominant content (despite multilingual data)
- Biases from the Wikipedia-based quality classifier (anything that "looks like Wikipedia" passes; content that doesn't — even if high quality — gets filtered)

The heavy filtering mitigates some issues but introduces others: the fastText classifier trained on Wikipedia references biases toward encyclopedic content, potentially undervaluing creative, conversational, or domain-specific text.

**11. Why does Wikipedia get 2.45 epochs but CommonCrawl only 1.10?**

Higher-quality data benefits from repetition because:
- The model can extract more from each example when the text is dense, factual, and well-structured
- Repetition of high-quality data acts as a form of emphasis — the model learns these patterns more strongly
- CommonCrawl is noisier, so repeating it amplifies noise alongside signal

However, there are limits: too much repetition causes memorization and reduces generalization. The paper doesn't discuss this tradeoff deeply, but subsequent work (e.g., Muennighoff et al., 2023, "Scaling Data-Constrained Language Models") found that data can be repeated up to ~4 epochs with continued improvement, after which returns drop sharply.

**12. Llama 2's pretraining data is undisclosed.**

Several possible reasons:
- **Legal risk:** Between LLaMA 1 (Feb 2023) and Llama 2 (Jul 2023), multiple lawsuits were filed over training data use. Being more specific about data sources increases legal exposure.
- **Competitive advantage:** The data mix is one of the few non-architectural differentiators between models.
- **Practical complexity:** The data pipeline likely evolved significantly and may be harder to describe cleanly.

This does undermine the "open" claim — you can reproduce the training code and use the weights, but you can't reproduce the data pipeline. Subsequent open-source efforts (e.g., RedPajama, DCLM) have tried to fill this gap by creating reproducible training datasets.

---

## Fine-tuning Pipeline (Llama 2)

**13. "Quality is all you need" — is 27k SFT examples really enough?**

Yes, for several reasons:
- The base model already has strong language capabilities — SFT doesn't teach language, it teaches *behavior*
- A few thousand examples are enough to establish the response format, tone, and basic instruction-following pattern
- Quality dominates because the model learns *style* and *patterns* from demonstrations, and inconsistent or low-quality demonstrations teach contradictory patterns

This finding was confirmed by subsequent work. Zhou et al. (2023, "LIMA") showed that even 1,000 carefully curated examples could produce a competitive chat model. The lesson: for behavior shaping (not capability building), data quality >> data quantity.

**14. Why does Llama 2 compute loss only on answer tokens?**

Computing loss on the full sequence (prompt + answer) trains the model to predict the prompt tokens, which is wasteful — the model already knows how to generate text like the prompt from pretraining. Worse, it can bias the model toward the distribution of prompts rather than answers.

Answer-only loss focuses all gradient signal on the behavior you want to change: how the model responds. This is more sample-efficient and produces cleaner behavioral changes. InstructGPT's approach of computing loss on the full demonstration may have included some prompt loss, but the paper isn't fully explicit. The answer-only approach became standard practice after Llama 2.

**15. Why two reward models instead of one?**

A single RM faces a fundamental tension: on safety-sensitive prompts, a helpful response might be unsafe, and a safe response might not be helpful. A single model must learn when to prioritize which dimension, adding complexity to the learning problem.

Two models sideline this tension:
- The Safety RM learns one clear objective: "is this response safe?"
- The Helpfulness RM learns a different objective: "does this response help?"
- The combining function ($R_c$) explicitly encodes the priority: safety first on flagged prompts, helpfulness otherwise

You could use three or more RMs (e.g., truthfulness, creativity, conciseness), but each additional RM adds training cost and combining complexity. The two-RM approach is a pragmatic sweet spot.

**16. What does the margin $m(r)$ in the RM loss actually buy you?**

Without the margin, "significantly better" and "slightly better" pairs provide the same gradient signal — the RM only learns that one is better, not *how much* better. With the margin, the RM learns calibrated scores: responses that are clearly better get clearly higher scores, while similar responses get similar scores.

This matters for downstream RLHF: a well-calibrated RM provides more informative reward signals. If the RM gives reward 5.0 to both excellent and mediocre responses (because it only learned ordering, not magnitude), PPO gets a less useful signal than if it gives 8.0 to excellent and 6.0 to mediocre. The paper reports this is especially helpful for the Helpfulness RM on samples where responses are clearly different.

**17. Iterative RLHF: why 5 rounds?**

The key problem is **distribution shift**: as RLHF improves the policy, its outputs become increasingly different from the outputs the RM was trained on. A RM trained on SFT outputs may not accurately score the much-better (or differently-bad) outputs of an RLHF-V3 policy.

Iterative RLHF addresses this by collecting new preference data from the latest model at each round, retraining the RM, and then running another round of RLHF. Five rounds likely represents the point of diminishing returns given the annotation budget. Each round requires collecting new human preferences (expensive) and retraining the RM. The decision to stop at V5 is pragmatic: they ran out of improvement headroom and/or annotation budget.

**18. Rejection Sampling vs. PPO — why use both?**

They have complementary strengths:
- **Rejection Sampling** explores broadly (K samples per prompt) but each sample comes from the same policy — no online learning
- **PPO** learns from each sample and updates the policy, building cumulative improvement — but only generates one sample per prompt

Using Rejection Sampling first creates a high-quality dataset (the best of K samples), which PPO then builds on. Think of it as: Rejection Sampling sets a strong starting point, PPO fine-tunes from there. The paper notes that until RLHF-V4, only Rejection Sampling was used; PPO was added later for additional gains. This suggests Rejection Sampling does most of the heavy lifting, with PPO providing marginal but meaningful improvements.

---

## Ghost Attention (GAtt)

**19. Why do RLHF models forget the system message?**

Attention-based models allocate "attention budget" based on what they've learned is important for predicting the next token. In multi-turn dialogue, the most recent turns are empirically most informative for the next response. The system message, being far back in the context, naturally receives less attention — especially after several turns push it further away.

Additionally, the RLHF training data may not consistently reinforce system message adherence across many turns, so the model doesn't learn that the system message should override recent conversational context. The model learns that recent turns > distant context, which is usually true for dialogue but not for system instructions.

**20. Is GAtt a hack or a principled solution?**

It's somewhere in between. GAtt is principled in that it explicitly addresses the training/inference mismatch: during training, the system message is only in the first turn; by training on data where the model must maintain the instruction, it learns the right attention pattern. The loss masking (zero loss on intermediate assistant messages) is a clean solution that avoids teaching the model to parrot the system message.

However, it is "hacky" in that it requires synthetic data generation with specific constraints, only works for instructions that can be concatenated to user turns, and has a finite horizon (the context window). A more principled approach might involve architectural changes to explicitly separate instruction-following from dialogue (e.g., separate instruction encoding paths), but GAtt's simplicity and effectiveness make it a pragmatic choice.

---

## Safety & Alignment

**21. Is 0% toxicity on ToxiGen actually good?**

0% on ToxiGen means the model has learned to avoid generating text that ToxiGen's classifier flags as toxic. This is a necessary but not sufficient condition for safety:
- The model might generate harmful content in ways the classifier doesn't detect (e.g., subtle manipulation, coded language, harmful advice that isn't "toxic")
- ToxiGen focuses on specific categories of toxicity — it doesn't cover all harmful outputs
- The model might be overly cautious, refusing legitimate requests to avoid any risk

That said, reducing measurable toxicity to near-zero is a significant achievement. The concern is confusing "passes this benchmark" with "is safe" — any benchmark measures a specific slice of the safety space.

**22. Context distillation: does it change "values" or just behavior?**

Context distillation changes the model's default behavior without changing its underlying capabilities. The model is still *capable* of generating unsafe content — it's just less likely to do so without a safety preprompt. This is more analogous to "behavioral training" than "value alignment."

An analogy: training a person to be polite doesn't eliminate their capacity for rudeness; it changes their default behavior. Similarly, context distillation shifts the default toward safety without eliminating the capability. This is why adversarial jailbreaks often work — they find ways to override the behavioral training and access the underlying capabilities.

**23. The false refusal problem.**

False refusal is likely inevitable to some degree with any safety tuning approach. The fundamental issue is that the model makes a binary decision (respond/refuse) based on a probabilistic assessment of safety, and any threshold will have false positives.

Approaches to minimize false refusal:
- Better calibration of safety scores (so borderline-safe prompts score higher than borderline-unsafe ones)
- Providing the model with the ability to ask clarifying questions instead of refusing
- Training on "borderline" examples that teach the model to distinguish "Christmas Crack" (candy) from actual harmful content
- Ensemble approaches that only refuse when multiple safety signals agree

**24. 350 red teamers — is that enough?**

Red teaming faces a fundamental scaling problem: the space of possible harmful interactions grows faster than the team can explore. 350 people is a large team by academic standards, but:
- They can only probe a fraction of the model's capability space
- Team composition matters — 350 similar people might miss things that 50 diverse people would catch
- As models get more capable, the attack surface grows (e.g., multi-step manipulation, combined-domain attacks)
- Red teaming is adversarial — the team is trying to break the model, but a determined attacker may be more creative

Red teaming is best viewed as one layer of defense, not a comprehensive safety solution. It catches obvious failures and common attack patterns but cannot guarantee safety against novel attacks.

**25. Safety data scaling: can you have too much safety data?**

The paper shows helpfulness staying constant as safety data increases, but this likely has limits:
- At extreme safety data ratios, the model may over-index on safety and become unhelpfully cautious
- The 0.05% false refusal rate is measured on a specific test set — it may be higher on real-world usage where prompts are more diverse
- There's a measurement problem: helpfulness tests may not capture subtle degradation (e.g., the model becoming more verbose, hedging more, or being less creative)

The paper's finding — safety can be added cheaply — is important and likely holds for moderate safety data levels. But the relationship probably isn't linear, and at extreme levels, helpfulness would eventually degrade.

---

## Evaluation & Benchmarks

**26. Human evaluation vs. model-based evaluation.**

Both have complementary strengths:
- **Human evaluation** captures nuance, can judge open-ended quality, reflects actual user preferences — but is expensive, noisy (low inter-rater agreement), and hard to scale
- **Model-based evaluation (GPT-4 as judge)** is cheap, consistent, scalable — but may have systematic biases, can be gamed, and can't evaluate capabilities beyond its own

The Llama 2 paper wisely uses both: model-based evaluation for rapid iteration (RLHF-V1 through V5), human evaluation for major version validation. This is sound practice. The concern about GPT-4 evaluating models that surpass it is real but not immediate — for now, GPT-4 is a reasonable proxy. As the gap narrows, model-based evaluation will need to evolve.

**27. Is "competitive with ChatGPT" meaningful?**

A 36% win rate with 31.5% ties means that in a random comparison, ChatGPT wins ~32.5% of the time and Llama 2-Chat wins ~36% of the time — it's roughly even with a slight edge to Llama 2-Chat on this specific test set. However:
- The test set (4k prompts) may not represent real-world usage
- ChatGPT was continuously updated — the comparison was against a specific snapshot (gpt-3.5-turbo-0301)
- Single-number comparisons hide domain-specific differences (ChatGPT may dominate on coding while Llama 2-Chat wins on dialogue)

"Competitive" is a reasonable framing — it means "close enough that the choice depends on other factors" (cost, privacy, customizability). It doesn't mean "equal" or "better."

**28. MMLU as the universal benchmark.**

MMLU has become the de facto standard, but it has limitations:
- It only measures multiple-choice knowledge, not open-ended generation, reasoning, or creativity
- Questions are extracted from existing exams, biasing toward academic knowledge
- It doesn't measure instruction following, safety, or conversational ability — all things users care about
- Saturation: as models approach 90%+, the benchmark loses discriminative power

The 18-point gap between Llama 2-70B (68.9) and GPT-4 (86.4) represents a significant difference in knowledge breadth and accuracy, roughly comparable to the difference between a strong undergraduate and a subject expert. However, MMLU differences may not map linearly to practical utility — a model at 69% may be perfectly adequate for most applications.

---

## Connections to Previous Weeks

**29. From GPT-3 to LLaMA: what changed and what stayed the same?**

The core architecture is remarkably similar — both are decoder-only transformers trained with next-token prediction on large web corpora. What changed:
- **Three small architectural improvements** (RMSNorm, SwiGLU, RoPE) — each individually minor but collectively meaningful
- **Training philosophy** — train smaller models on more data rather than bigger models on less data
- **Data curation** — much more emphasis on filtering and deduplication
- **Open release** — perhaps the biggest difference, enabling community-driven improvement

LLaMA is not qualitatively different from GPT-3 — it's the same paradigm executed more efficiently with better engineering practices. The innovations are primarily in training strategy, not architecture.

**30. From InstructGPT to Llama 2-Chat: is this progress or iteration?**

Both. The core pipeline (SFT → RM → PPO) is identical. The additions are meaningful but incremental:
- Two RMs instead of one
- Margin-based loss
- Iterative RLHF
- Rejection sampling
- Ghost Attention
- Safety-specific pipeline

A truly different approach would be something like DPO (no reward model, no RL), Constitutional AI (AI-generated feedback), or debate-based alignment. Llama 2-Chat shows that InstructGPT's basic framework, when carefully scaled and iterated, can produce competitive results.

**31. The full arc: what was the single most important advance?**

This is deliberately open-ended, but reasonable arguments exist for each:
- **Transformers (W2):** Without the architecture, nothing else is possible. But transformers without scale or alignment would be academic curiosities.
- **Scale (W1, GPT-3):** Showed that scale unlocks emergent capabilities (few-shot learning). But unaligned scale produces capable but unreliable models.
- **Alignment (W4, InstructGPT):** Made language models actually *useful*. A 1.3B aligned model beats a 175B unaligned one.
- **Open release (W5, LLaMA):** Democratized access and accelerated the entire field. Without LLaMA, the open-source ecosystem would be years behind.

A case can be made that alignment (W4) was the most important because it solved the *right problem* — bridging the gap between what models *can* do and what users *need* them to do.

---

## Broader Questions

**32. Is the open-source model gap closing?**

As of early 2026, the gap has narrowed significantly. Llama 3 and subsequent open models have continued to improve, and on many practical tasks, open models are comparable to closed frontier models. However, the frontier (GPT-4.5/Claude 4.5 class) still leads on the hardest reasoning and multimodal tasks. The gap is smallest for well-defined tasks (coding, Q&A) and largest for complex reasoning and novel problem-solving.

**33. What did Llama 2 get wrong?**

In hindsight:
- **34B model was delayed** — it was held back from release due to insufficient red teaming, suggesting the safety pipeline wasn't scalable enough
- **RLHF complexity** — the 5-iteration pipeline with two RMs, rejection sampling, and PPO is expensive and hard to reproduce. Later work showed that simpler approaches (DPO, ORPO) can achieve similar results
- **Toxicity benchmark choice** — ToxiGen has been criticized for inconsistent scoring; achieving 0% may partly reflect benchmark limitations rather than true safety
- **Context length** — 4k context was already feeling limited by mid-2023; later models moved to 128k+

**34. RLHF vs. DPO vs. other alignment methods.**

- **RLHF (Llama 2):** Full pipeline — train RM, then RL. Pros: most studied, flexible reward shaping. Cons: complex, unstable training, requires separate RM.
- **DPO:** Skip the RM entirely — directly optimize the policy on preference pairs using a modified cross-entropy loss. Pros: simpler, more stable, no RM needed. Cons: less flexible (can't shape rewards), may underperform RLHF at scale.
- **Constitutional AI (Anthropic):** Use the model itself to generate feedback, replacing human annotators. Pros: cheaper, scales with compute. Cons: may reinforce model biases.
- **ORPO, KTO, IPO, etc.:** Various simplified alternatives with different tradeoffs.

Meta likely stuck with RLHF because it was well-understood, already showed strong results with InstructGPT, and the team had engineering infrastructure for it. DPO was published concurrently (May 2023) and hadn't been validated at Llama 2's scale.

**35. The emergence question.**

The temporal perception and tool use findings are intriguing but should be interpreted cautiously:
- **Temporal perception:** The model was given 1,000 time-focused SFT examples — this is not zero-shot emergence but rather few-shot generalization from a small seed
- **Tool use:** The model likely encountered API-calling patterns in its pretraining data (documentation, tutorials, etc.) and RLHF amplified this capability

"Emergence" here means "a capability that becomes more reliable after alignment, despite not being explicitly trained for." This is plausible — RLHF may surface capabilities that exist in the pretrained model but aren't reliably triggered. Whether to call this "emergence" or "capability elicitation" is partly a definitional question.
