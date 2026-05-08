# Week 9 — Discussion Questions & Suggested Answers
**Paper(s):** (1) Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models, Cheng et al. 2026 (Engram); (2) DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence, DeepSeek-AI 2026

These are suggested answers to guide discussion, not definitive answers. Many of these questions are deliberately open-ended.

---

## Engram: Architecture and Lookup Mechanism

**1. What does "conditional memory" actually buy that MoE doesn't?**

The irreducible difference is **deterministic vs. learned routing**. MoE's router is a learned function of the hidden state, so the "address" of an expert depends on intermediate computation — making prefetching impossible and forcing experts to live on-device. Engram's hash address is computable from input tokens *before* any forward pass executes. This is what enables the system co-design (PCIe prefetching, host DRAM tables, NVMe spillover for the Zipfian tail) that V3-class MoE simply cannot do. A second difference is *capacity efficiency*: an MoE expert is a full FFN block (parameters $\sim d^2$), while an Engram lookup is a single $d_\mathrm{mem}$-dimensional vector. So at fixed parameter budget, Engram concentrates parameters in *atomic* memory units rather than full computational units. The U-shape result implies these capacity types are *complements*, not substitutes — neither can fully replace the other.

A more theoretical answer: MoE's routing must distinguish hidden states (which is a continuous, unbounded space), while Engram's hashing operates on tokens (a finite, discrete space). The two routing problems are fundamentally different — and Engram exploits the discrete structure that MoE has to relearn.

**2. Why exactly $\{2, 3\}$-grams and 8 hash heads?**

The ablation shows adding 4-grams *hurts* val loss under fixed budget. The likely reason is dilution: the 1.6B-parameter Engram budget gets split among more N-gram orders. Since 4-grams are dramatically rarer than bigrams/trigrams in a Zipfian distribution, most of the 4-gram capacity sits unused while 2/3-gram capacity is over-pressured. Under a *much larger* memory budget (e.g., $> 50$B), 4-grams might pay off because there's enough capacity to store the long tail of higher-order patterns. The paper hints at this: "we do not rule out that higher-order N-grams become beneficial at larger memory scales."

8 hash heads is a collision-vs-capacity tradeoff. With $K$ heads and $M$-prime tables, the expected fraction of "fully colliding" entries (same hash in all $K$ heads) is roughly $1/M^{K-1}$ for a single pair. For $M \approx 10^6$ and $K = 8$, this is astronomically small. But each head also costs $d_\mathrm{mem}$ parameters per slot, so $K$ trades off against the number of distinct slots. 8 likely sits near the empirical sweet spot.

**3. Hash collision tolerance.**

For the 5.7B Engram-27B at $d_\mathrm{mem} = 1280$, the implied slot count is $\sim 5.7 \times 10^9 / (8 \times 2 \times 1280) \approx 2.78 \times 10^5$ slots per (head, order) bucket. With 16 buckets total (2 orders × 8 heads), that's ~$4.5 \times 10^6$ slots overall. For two distinct N-grams to collide in *all* 8 heads (within one order), the joint probability is $(1/M)^7$ (assuming independent hashes), which for $M = 2.7 \times 10^5$ is $\sim 10^{-39}$ — effectively impossible.

In practice, Engram is robust to *partial* collisions: the gating mechanism can suppress noise on a per-row basis, and the multi-branch architecture (4 branches with separate Key projections) gives the model multiple chances to route the right signal. The pathological full-collision case is so rare it doesn't matter at this scale — but the paper doesn't analyze this directly, which is a small gap.

**4. Walk through the gating equation.**

RMSNorm on both query and key prevents magnitude blow-up — without it, hidden states $\mathbf{h}_t$ that happened to have larger norms would dominate the gate regardless of semantic alignment. Normalizing makes the dot product strictly an *angular similarity*. This is also necessary for gradient stability: if $\mathbf{h}_t$ is at the start of a Transformer layer where activations are relatively well-controlled, but the gating sigmoid pushes its output close to 0 or 1, gradients through it collapse. Pre-normalization keeps the gate operating in its sensitive regime.

A scalar $\alpha_t$ rather than softmax-over-rows is appropriate because Engram already retrieves *one* row per (head, order) and concatenates — there's no need to softmax-select among multiple rows. The retrieval has already committed to this set; the gate decides whether to *trust* it as a whole.

If $\alpha_t$ were per-head, the system would gain finer modulation — different attention heads could decide independently whether the lookup is useful. This is exactly what the multi-branch mHC integration *does*: each branch has its own Key projection $\mathbf{W}_K^{(m)}$, producing independent gates per branch. So the paper has effectively chosen "per-branch gating" as a softer alternative to per-head gating.

---

## Engram: The U-Shape Scaling Law

**5. Interpret the U-shape robustly.**

The $\rho \approx 75\text{–}80\%$ optimum is shown stable at sparsity ratio $\sim 10$ ($P_\mathrm{tot}/P_\mathrm{act}$). At lower sparsity (e.g., 4 — fewer experts, less inactive budget), the absolute value of the gains shrinks because there's less freedom to reallocate. At higher sparsity (20+), the gains might grow but the optimum could shift — perhaps Engram's value is most useful when expert capacity has hit diminishing returns. The paper doesn't sweep this dimension, so we should be cautious about generalizing the specific 75-80% number to other sparsity regimes. It's plausible the U-shape *exists* in all regimes but with different optimal points.

**6. Iso-FLOPs vs. iso-loss comparisons in Section 5.**

The iso-loss comparison (Engram-27B-46k vs. MoE-27B-50k, both at val loss 1.63) is an honest attempt to control for "better base model." But there are subtle confounds: (a) Engram's better val loss earlier in training may reflect *different learning dynamics* — the model may have absorbed the curriculum differently. (b) YaRN extension is sensitive to the base model's frequency-band utilization, which depends on how the model uses attention — and Engram's attention behavior differs from MoE's. A cleaner control would be to extend the same model with multiple post-pretraining strategies (YaRN, ALiBi, ABF) and verify the gains hold across all, ruling out a YaRN-specific interaction.

**7. The "infinite memory regime" log-linear fit.**

Log-linear over a 1.5-decade range ($2.6 \times 10^5$ to $10^7$) is suggestive but not conclusive. By analogy to the OverEncoding baseline that *does* saturate by similar scales, Engram's log-linear fit could break around $10^9 \text{–} 10^{10}$ slots — beyond which the long-tail N-grams have such low frequency that they're never seen during training. The asymptotic ceiling is set by the entropy of natural-language N-grams: there's a finite amount of useful pattern mass to memorize, and once Engram has captured the head and middle of the Zipfian distribution, additional capacity has nothing to learn.

If scaling continues, it would imply natural language has surprisingly heavy-tailed compositional structure (more useful unique N-grams than we'd assume), or that the 262B-token training corpus is the bottleneck rather than the architecture.

---

## Engram: Mechanistic Claims

**8. The "effective depth" claim.**

CKA similarity is necessary but not sufficient for the depth claim. Two layers with similar representations *could* be doing fundamentally different computations — CKA can't distinguish "same output, different processing" from "same depth-equivalent." A cleaner test would be: (a) **circuit-level interpretability** — find a known computation (e.g., subject-verb agreement) and trace where in the Engram-27B layer stack it's resolved vs. MoE-27B. If Engram resolves it 7 layers earlier, the depth claim is supported. (b) **Intervention experiments** — patch the activation of Engram-27B layer 5 with the activation of MoE-27B layer 12 and see if downstream performance is preserved. This is the standard test for representational equivalence in interpretability research.

**9. Sensitivity analysis dichotomy.**

The retention pattern is more nuanced than a clean factual / contextual split:
- **Pure factual**: TriviaQA (29%), PopQA (44%), TriviaQA-ZH (44%) — collapse confirms factual specialization.
- **Algorithmic / math**: GSM8K (44%), MATH (36%), MGSM (44%) — collapse shows Engram contributes to *number / formula manipulation*, not just facts. This is consistent with Engram learning common math expressions ("a^2 + b^2", "$\int_0^1$") as N-gram patterns.
- **Code**: HumanEval (62%), MBPP (68%) — partial collapse, suggesting Engram captures common code patterns (function signatures, idioms).
- **Reading comprehension**: 81-93% — preserved because attention is doing the heavy lifting of context grounding.

So the actual specialization is **stereotyped patterns** (broadly construed: facts, math expressions, code idioms) — not just factual knowledge. The paper's framing slightly under-sells this.

**10. Gating visualization (Figure 7).**

The activation pattern — gate fires on the *completed* multi-token entity, after the last token is consumed — supports the interpretation that Engram is functioning as a "completion-of-pattern" detector. The mechanism: the suffix N-gram $g_{t,n}$ ends at position $t$; if $t$ is the last token of "Diana, Princess of Wales", then the trigram retrieves the entity-level memory exactly when needed. This is closer to *post-hoc entity recognition + memory fetch* than to *prediction-of-next-token-from-context*. Functionally, the model uses the lookup to answer "what's the canonical representation of this completed entity" rather than "what should come next given partial context."

This is consistent with the LogitLens result — Engram representations are more "prediction-ready" earlier in the network because the lookup has already canonicalized the entity, and downstream layers can directly use the canonical form.

---

## DeepSeek-V4: Architecture Choices

**11. CSA vs. HCA — why both?**

CSA captures **content-selective long-range** dependencies: top-$k$ over compressed blocks lets each query pick the most relevant compressed entries from anywhere in context. HCA captures **coarse global summarization**: dense attention over a much shorter ($\sim n / 128$) sequence forces the model to maintain a coarse summary of the entire context that every query attends to.

A single design with adaptive $m$ wouldn't capture this duality cleanly — a moderately-compressed dense attention is neither content-selective enough (dense forces uniform attention to all entries) nor coarse-enough (small compression rate doesn't summarize aggressively). The interleaving means alternating layers focus on different scales of context dependency, and cross-layer composition lets the model assemble queries that benefit from both. From an engineering side, interleaving also balances the KV cache load across layers — every layer has its own cache size, distributing memory pressure rather than concentrating it.

**12. mHC's manifold constraint.**

The Sinkhorn-Knopp iteration adds 20 small (size $n_\mathrm{hc} \times n_\mathrm{hc} = 4 \times 4$) matrix operations per layer per forward — negligible per-step compute (rounding error vs. attention). The doubly-stochastic prior is right because: (a) it bounds spectral norm by 1 (so signal can't blow up across deep stacks), (b) it preserves *information flow* (each output dimension is a convex combination of input dimensions, no information is "thrown away" or "infinitely amplified"), (c) it's closed under multiplication, so chain composition stays in the manifold.

Orthogonal matrices would also bound spectral norm but allow negative entries (information cancellation between branches). Symmetric PSD would require a different structural meaning. The doubly-stochastic constraint specifically captures "non-cancellative redistribution of branch contributions," which is what a residual stream needs.

**13. Hash routing for the first 3 MoE layers.**

Bottom-of-stack representations are still primitive — token embeddings haven't yet developed the rich structure that learned routing can exploit. Random-routing or hash-routing at this stage avoids the early-training instability where the router's decisions oscillate based on poorly-formed hidden states. Hash routing is deterministic so it's also implicitly *load-balanced* across batches (assuming a uniform hash distribution).

Extending to 6 hash-routed layers would push the deterministic regime higher in the stack. The cost is loss of routing flexibility — the model can no longer learn to route similar tokens to similar experts at those layers, so semantic specialization in the early stack is lost. The empirical 3-layer choice is likely the sweet spot found by ablation. Whether it generalizes to deeper models is open.

**14. Anticipatory Routing's $\Delta t$.**

The paper bounds $\Delta t$ by ~20% wall-clock overhead, suggesting $\Delta t$ is small (one or two steps). The upper bound comes from divergence: if $\Delta t$ is too large, routing is operating on a stale model snapshot, which itself becomes a source of drift. The lower bound is driven by the sufficient-decoupling requirement.

This is *probably* a temporary fix. The fundamental issue is that MoE training has positive feedback loops where bad routing decisions compound. Better optimizer design (e.g., Muon variants), routing regularizers, or even fundamentally different routing mechanisms (hash, learned-but-fixed-after-pretraining) might eliminate the need for Anticipatory Routing. But for now, it's a pragmatic and effective patch.

**15. SwiGLU clamping.**

The clamping range $[-10, 10]$ is wide relative to typical pre-activations under healthy training (which would be in $[-3, 3]$ or so). So clamping affects only the outlier tails — the ~0.1% of activations that have grown anomalously large. These outliers correspond to "broken" expert weights or routing decisions that produce extreme values; clamping prevents them from cascading into loss spikes through the softmax in attention or downstream MoE routing.

This trick is general — it works because the *useful* signal in SwiGLU activations sits well below 10, and any value above that is almost certainly noise. It's not Muon-specific, but it interacts well with Muon because the orthogonalized updates produce more uniform-magnitude weight changes, leaving outliers as cleaner anomalies that clamping can target.

---

## DeepSeek-V4: Training and Post-Training

**16. OPD vs. Multi-Domain RL.**

Multi-domain RL trains specialists *jointly*, sharing parameters. This causes interference: gradient updates from a math RL signal can unlearn coding capability, because both compete for the same network's representations. OPD avoids this by training specialists *independently* and then distilling — the student receives clean, non-conflicting per-domain teaching signals. Conceptually, OPD is closer to mixture-of-experts at the *training* level: each domain has its specialist; the unified student "merges" them via distillation.

When would joint RL beat OPD? If domains share substantial *transferable* skill — e.g., logical reasoning underlying both math and coding — joint RL can amplify this transfer in ways OPD cannot. The bet OPD makes is that domain-specific reward signals are sufficient and that the cross-domain transfer is small. For a model targeting *agentic generalists*, joint RL might be worth re-investigating.

Credit assignment in multi-domain RL: with sparse outcome rewards, the gradient signal is noisy. When this noise is shared across domains, learning is unstable. Specialists + OPD removes the noise problem by making each gradient computation single-domain.

**17. Full-vocabulary KL vs. token-level KL estimator.**

The token-level estimator uses $\log(\pi_E(y_t)/\pi_\theta(y_t))$ as a per-token advantage. This is a **single-sample estimate** of the per-token KL, computed only at the realized token from a single rollout. It's high-variance because the realized token is a draw from the *student's* distribution, while we're estimating the *teacher's* full distribution. Many tokens at low teacher probability are missed entirely.

Full-vocabulary KL exploits the entire output distribution — it's an exact computation, not a Monte Carlo estimate. Lower variance, faster convergence, more faithful distillation. The trade-off is memory: for $|V| = 128K$ and 10 teachers, naive logit materialization is prohibitive. V4 caches teacher hidden states + reconstructs logits on-the-fly, so memory cost is bounded by the prediction head size. This is feasible only because the prediction head is small relative to the rest of the model.

In RLHF, token-level estimators are preferred because the "teacher" is a reward model (not a logit distribution), so full-vocab KL doesn't apply. Distillation is structurally different.

**18. FP4-to-FP8 lossless dequantization.**

FP8-E4M3 dynamic range is roughly $2^{-9}$ to $2^{8}$ (about 7 orders of magnitude). FP4-E2M1 dynamic range is roughly $2^{-1}$ to $2^{1}$. The condition for lossless dequant: within an FP8 quantization block (128×128 tile), the ratio max/min FP4 sub-block scales (1×32 tiles) must fit within FP8's dynamic range. Empirically, the paper reports MoE expert weights satisfy this — meaning the per-tile scale variance within each block is small enough.

Could a future model violate it? Yes, plausibly — if a model has highly heterogeneous expert weight magnitudes (e.g., some experts barely used, others heavily specialized), the scale ratio could exceed the threshold. V4's expert weights apparently don't exhibit this, perhaps because of the auxiliary-loss-free balancing, but it's not a guarantee.

**19. The 1.6T-parameter trillion-club bet.**

Both bets are reasonable, but they target different deployment profiles:

| Approach | Sweet Spot |
|----------|-----------|
| Sparse MoE (V4-Pro 1.6T/49B) | Cloud serving with batched, multi-tenant inference. Activated parameters dominate per-token compute; total parameters become a memory cost amortized across many requests. |
| Dense (Llama 3 405B) | Single-stream low-latency inference, edge / on-prem deployments without expert-parallelism infra, simpler deployment pipelines. |

V4-Pro's 49B activated is reasonable for a single H100 GPU's compute ceiling; the 1.6T total requires ~3-4 H100 GPUs of memory but is fixed amortizable cost across requests. Llama 3 405B requires ~8 H100s for memory and engages all of them per token.

Which is more compelling depends on perspective. For research / open-weights deployment by individuals and small labs, Llama 3's simplicity is a real advantage. For frontier-scale serving (OpenAI, Anthropic, DeepSeek's own production), MoE wins.

**20. "Think Max" mode.**

Honest answer: probably some of both. The Think Max system prompt is an *instruction-following* surface that the model has been RL-trained to respect. The longer reasoning traces it produces likely do contain genuine extra computation (Toggle's Phase 1 standard scaling phase explicitly trains the model to leverage extra tokens for accuracy), but they also contain stylistic markers of carefulness ("let me consider all paths", "stress-testing this hypothesis") that may be performative.

Distinguishing experiments: (a) Compare Think High vs. Think Max on benchmarks where the *correct* approach requires structured exhaustive search vs. clever insight. If Think Max wins on the former and ties on the latter, it's adding genuine search depth. (b) Inject the Think Max system prompt into a shorter token budget — does Think Max-style reasoning still produce gains, or does it degrade because the budget can't sustain the imitated form?

---

## DeepSeek-V4: Long Context

**21. The 1M-token claim, in practice.**

"Natively supports 1M" matters even with 0.59 MMR at the extreme, for several reasons: (a) most real workloads don't need 1M — they need 200-500K (long codebases, scientific paper sets, conversational histories). The model is *useful* at 512K (0.85 MMR), even if degraded at 1M. (b) "Natively supports" is contrast to models that simply *fail* past 128K; partial success matters. (c) The KV cache cost reduction (V4-Pro's 10% of V3.2's at 1M) makes 1M serving economically viable, which incentivizes building applications that test the limit.

Useful 200-500K scenarios: full-codebase code agents (mid-size repos), legal contract analysis with multiple referenced documents, scientific literature review across 30-50 papers, long agentic workflows accumulating tool history.

**22. CSA top-$k$ choice (512 for Flash, 1024 for Pro).**

Top-$k$ scales sublinearly with context — at 1M tokens compressed to 250K blocks (with $m = 4$), top-1024 attends to ~0.4% of compressed blocks, vs. attending to ~12% at 32K. Information-theoretically, the right top-$k$ is the rank at which the signal-to-noise ratio of "relevant compressed blocks" tapers. For natural language, the relevant fraction grows sublinearly with context (more haystack, but proportionally fewer needles), so a fixed top-$k$ might actually become *more* selective at longer contexts.

At 10M, top-2048 might be appropriate. There's no fundamental information theoretic argument — it's a noise-floor calibration, and depends on the specific compression scheme's lossiness.

**23. Sliding window $n_\mathrm{win} = 128$.**

128 likely matches a typical "short-range context" length — enough to capture immediate sentence and clause structure, where compression's loss is most painful. The interaction with $m$ and $m'$: since $m = 4$, the sliding window covers the last 32 *compressed* blocks, providing dense detail for the most recent tokens. With $m' = 128$, the sliding window covers the last *single* HCA-compressed block, again ensuring the most recent compressed entry has high-fidelity backup.

Sensitivity: smaller $n_\mathrm{win}$ would risk losing immediate-token detail; larger would inflate KV cost without proportional benefit (compression already captures medium-range). 128 is a reasonable middle.

---

## Cross-Paper / Shared Primitives

**24. mHC and Muon shared between papers.**

The shared primitives signal a coordinated *DeepSeek architectural research line*. Both papers cite Xie et al. 2026 for mHC and Liu et al. 2025 for Muon — these are likely internal-then-public publications from a tightly coordinated team. This pattern (architectural primitives developed in dedicated papers, then composed into flagship model papers) is similar to how Google's PaLM / Gemini line shared primitives with concurrent papers on chinchilla scaling, MoE techniques, etc.

This bodes well for future open releases — if mHC and Muon are foundational primitives, they'll appear in upcoming DeepSeek-V4.5, DeepSeek-R2, and likely third-party adoptions (Kimi K2.5 already uses MuonClip). The composability suggests the team is building toward a *modular foundation* where each new model recombines proven primitives.

**25. Two complementary sparsity axes.**

Combining Engram's lookup with V4's CSA/HCA could yield a model that is sparse in *three* axes: routed experts, embedded knowledge lookup, and compressed attention. The expected gains would compound on tasks needing all three (e.g., long-context factual QA: long context handled by CSA, fact retrieval handled by Engram, reasoning by MoE). The specific gain wouldn't simply add — they multiply across orthogonal dimensions of capability.

Engineering challenges: training stability of the combined system, alignment of MoE expert specialization with Engram's lookup behavior, and managing three distinct memory hierarchies (HBM for attention, compressed cache for KV, host DRAM for Engram).

**26. OPD vs. Engram for knowledge.**

V4's approach: train on more data, use OPD to consolidate domain-specialist knowledge into one student. Knowledge is **diffuse in parameters** — encoded across all weights through SGD optimization on pretraining data + specialist supervision.

Engram's approach: bake *static* knowledge into a dedicated lookup table, leaving the backbone for dynamic reasoning. Knowledge is **concentrated in addressable memory** — encoded as vectors retrievable by hash.

Generalization to new domains: V4-style is more flexible (continued pre-training adds knowledge naturally) but expensive (every new domain requires expensive re-training). Engram-style is more compositional (potentially: just augment the lookup table with new entries for the new domain) but requires careful integration with the trained backbone — naive table augmentation might break gating.

In the long run, the two are likely composable: bake static, hash-addressable factual knowledge into Engram, use V4-style training for the dynamic reasoning capability that consumes those facts.

---

## Connections to Previous Weeks

**27. Mixtral (W6) → DeepSeekMoE → Engram + V4.**

Mixtral (W6): 8 experts, top-2. Demonstrated that MoE works at scale with simple top-$k$ routing and a learned router. *Necessary first step* — proved MoE viability.

DeepSeekMoE: fine-grained experts (256 vs. 8), shared experts (always-on), auxiliary-loss-free balancing. *Necessary refinement* — solved instability and load-imbalance problems that plagued early MoE.

Engram-27B: reallocates expert capacity to memory. *Contingent extension* — demonstrates that the sparse capacity from MoE can be reallocated to a different sparsity primitive. Not strictly necessary, but unlocks new system-level efficiency.

V4-Pro: combines DeepSeekMoE with hash routing for early layers + Anticipatory Routing for stability + 1.6T scale. *Necessary scale step* — demonstrates that the recipe holds at $> 1$T parameters.

The lineage is: MoE viability → MoE refinement → MoE composability with new primitives → MoE at extreme scale.

**28. InstructGPT (W4)'s SFT+RLHF → V4's specialist+OPD.**

InstructGPT: SFT on demonstrations → reward model on preferences → PPO RL with the reward model. Single pipeline, single objective.

V4: SFT on domain data (per specialist) → GRPO RL with rule-based + GRM rewards (per specialist) → On-Policy Distillation merge.

Mapping:
- SFT in both is similar (demonstration data, supervised learning).
- InstructGPT's reward model → V4's GRMs (still trained reward signals, but now generative and rubric-based).
- InstructGPT's PPO RL → V4's GRPO (V4 uses group-relative advantages, InstructGPT uses learned value baseline).
- *Novel in V4*: domain specialist parallelism + OPD merge (no analog in InstructGPT — they trained one general model).

The biggest structural change is the *ensemble-then-distill* paradigm, which InstructGPT didn't anticipate.

**29. Mistral 7B SWA (W6) → V4 sliding-window branch.**

Mistral 7B's SWA: $W = 4096$ window only. Information propagates across the full context only through *layer composition* — token $t$ at layer $k$ can attend to tokens up to $kW$ behind. This is a clean, FLOPs-cheap design but loses fine resolution at long range.

V4's SWA branch (window 128) is orthogonal: it provides *local detail* for the most recent 128 tokens, while CSA/HCA handle long-range. The community moved from "use SWA alone for everything" (Mistral) → "use compression for long-range, supplement with small SWA for local" (V4). The change in understanding: long-context retrieval *cannot* be fully reconstructed from layer-composed local attention alone — it needs an explicit long-range mechanism (compression / sparse). SWA's role shrunk from "the attention design" to "the local-detail supplement."

**30. DeepSeek-R1 (W8)'s pure RL → V4's specialist + OPD.**

R1 (Pure-RL): one base model, one RL signal. Demonstrated emergent CoT, but suffers from credit-assignment noise across domains.

V4 (Specialist + OPD): one base model, many specialists trained with domain-specific RL, then merged. The structural advantage is *clean credit assignment per domain*.

The shift suggests the field has learned: pure RL on a strong base produces emergent reasoning, but it doesn't *control* what the model gets good at. Specialists + OPD lets you steer each domain's learning explicitly. This is a more engineered approach — less "miracle of RL," more "engineered learning pipeline."

That said, R1's lessons aren't abandoned. V4's GRPO (specialist phase) is exactly R1's RL recipe, applied per-domain.

---

## Broader Questions

**31. The compute-vs-storage tradeoff at scale.**

Engram and V4 are *complementary*, not competing. Engram's argument: store more parameters off-chip, retrieve them via deterministic hashing. V4's argument: compress per-token compute by reducing the working set.

Combined: Engram's deterministic retrieval works *because* the working set per layer is small (V4-style compression). V4's compression works *because* the model has enough capacity (Engram-style memory) to encode the long-tail static knowledge that compressed attention would otherwise lose. The two visions are likely combined in the next generation — DeepSeek's recipe is "more capacity, less compute, smarter retrieval."

**32. Open-source Frontier Gap.**

The 17-point SimpleQA-Verified gap to Gemini-3.1-Pro likely comes from three places:
1. **Knowledge data quality**: Gemini has decades of Google search/index access for fact-quality curation.
2. **Knowledge retrieval reliability**: at inference, Gemini can blend internal knowledge with live retrieval; V4's open weights cannot.
3. **Targeted post-training for factual verification**: Gemini likely has dedicated factuality post-training, possibly with specialized retrieval-augmented training.

Bottleneck is likely (1) → (3). A well-resourced open-source effort could match (3) and partially close (2) via toolkit integration, but (1) is hard to compete with without Google-scale data access.

**33. Cost of the Trillion-Parameter Club.**

Justifying scenarios for V4-Pro 1.6T: (a) high-leverage agentic workflows (long-running, multi-tool, complex) where each token's quality matters more than throughput. (b) Model-as-a-Service with batched, multi-tenant inference where activated cost (49B) dominates and total parameters amortize. (c) Tasks that benefit from broad knowledge (SimpleQA, broad multilingual) where V4-Pro's 33T pre-training tokens pay off.

V4-Flash (284B / 13B activated) is likely the more practical real-world target for most deployments. It serves the broader market with substantially lower memory footprint (~1/6 of V4-Pro) while preserving most capability. The 50B-100B activated range tends to be the sweet spot for production inference cost vs. capability.

**34. What's the next axis?**

A few candidates:
- **Conditional retrieval** (RAG-as-architecture): V4 already partially blends retrieval into long-context attention. Native RAG with RL-trained retrieval policies could eliminate the inference-time RAG/fine-tune dichotomy.
- **Conditional adaptation** (per-token LoRA / adapter selection): each token activates different adaptation deltas based on local context. Builds on MoE's conditional computation but at a finer granularity.
- **Conditional decoding**: different decoding strategies (greedy, beam, sampling, tree search) per token based on uncertainty / task type. K2.5's Toggle is a coarse version of this.
- **Continual learning**: a reliable mechanism for absorbing new knowledge without retraining. Engram's static lookup gestures toward this — if you can update the table, you have a continual-learning primitive.

The next big jump likely needs to combine *several* of these axes into a unified architecture. The "single architectural breakthrough" era may be over; we're entering an era of careful composition.
