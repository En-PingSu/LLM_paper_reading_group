# Week 9 — Discussion Questions
**Paper(s):** (1) Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models, Cheng et al. 2026 (Engram); (2) DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence, DeepSeek-AI 2026

---

## Engram: Architecture and Lookup Mechanism

1. **What does "conditional memory" actually buy that MoE doesn't?** The paper frames Engram as a *new axis of sparsity*, complementary to MoE's conditional computation. But MoE experts can also memorize stereotyped patterns. What's the irreducible difference between activating a routed expert and looking up an embedding row? Why couldn't MoE — with enough capacity and an appropriate router — recover Engram's gains?

2. **Why exactly $\{2, 3\}$-grams and 8 hash heads?** The ablation in Figure 5 shows that adding 4-grams *hurts* val loss under a fixed memory budget. Walk through why: is it dilution of the most frequent bigrams/trigrams, or is something more fundamental going on? Under what scaling regime might 4-grams help?

3. **Hash collision tolerance.** Multi-head hashing uses $K = 8$ heads to mitigate collisions. Estimate the expected collision rate for a specific N-gram in a $5.7$B-parameter table, given the prime-sized hash tables. How robust is the model when two semantically distinct N-grams hash to the same row in *all* $K$ heads (a pathological but possible event)?

4. **Walk through the gating equation.** Why is RMSNorm applied to both query and key before the dot product? Why a sigmoid scalar gate $\alpha_t$ rather than a softmax over multiple memory rows? What would change if $\alpha_t$ were learned per-head instead of single-scalar?

---

## Engram: The U-Shape Scaling Law

5. **Interpret the U-shape robustly.** The paper shows the U-shape across two compute regimes ($2 \times 10^{20}$ and $6 \times 10^{20}$ FLOPs) at sparsity ratio $\approx 10$. Does the optimum $\rho \approx 75\text{–}80\%$ depend on the chosen sparsity? What if you're at $P_\mathrm{tot}/P_\mathrm{act} = 4$ vs. 20? The paper claims "robust" but only spans one ratio.

6. **Iso-FLOPs vs. iso-loss comparisons in Section 5.** Engram-27B at 46k steps matches MoE-27B's loss but uses fewer training FLOPs and beats it on long context. This separates *architectural superiority* from *better base model.* But could the architecture be exploiting an idiosyncrasy of YaRN extension — e.g., better-distributed RoPE-relevant features — that's specific to this protocol? How would you control for that?

7. **The "infinite memory regime" log-linear fit (Figure 3 right).** Engram scales log-linearly with embedding slots up to $10^7$ slots (~13B added params). At what slot count would you predict diminishing returns? If the scaling continues indefinitely, what does that imply about the upper bound of useful N-gram-level memorization in language data?

---

## Engram: Mechanistic Claims

8. **The "effective depth" claim.** CKA analysis shows Engram-27B's layer 5 aligns with MoE-27B's layer 12. The paper interprets this as a $\sim 2.4\times$ increase in effective depth in early blocks. But CKA measures *representational similarity*, not computational depth — could two layers be similar without one being a "deeper-equivalent" of the other? What would be a cleaner test of the depth claim?

9. **Sensitivity analysis dichotomy.** When Engram is silenced at inference, factual benchmarks collapse to 29% retention but reading comprehension preserves 81%+. The paper presents this as evidence of "factual specialization." However, GSM8K (44%) and MATH (36%) — which are *not* purely factual — also collapse. What's the actual specialization pattern?

10. **Gating visualization (Figure 7).** The gate $\alpha_t$ fires strongly on *completed* multi-token entities ("Alexander the Great", "Princess of Wales") — the N-gram is consumed, then the gate activates on the next token. What does this say about *when* in the linguistic processing the lookup contributes? Is Engram acting more like a "completion of pattern" detector or a "knowledge fetch upon entity recognition"?

---

## DeepSeek-V4: Architecture Choices

11. **CSA vs. HCA — why both?** V4 interleaves CSA ($m=4$, sparse top-$k$) and HCA ($m'=128$, dense). Walk through what each captures that the other misses. Could a single design with adaptive compression rate ($m$ chosen per layer) substitute for both? What's the engineering reason to interleave them rather than stack one then the other?

12. **mHC's manifold constraint.** The Birkhoff-polytope projection of $B_\ell$ is enforced via 20 Sinkhorn-Knopp iterations — added to every forward pass. What's the per-step compute overhead? Why is the doubly-stochastic constraint specifically the right structural prior here, vs. e.g., orthogonal matrices or symmetric positive-semi-definite?

13. **Hash routing for the first 3 MoE layers.** This is a deterministic-routing patch on an otherwise-learned-routing system. What problem does it solve at the bottom of the stack that learned routing doesn't? Could this idea be extended further — would 6 hash-routed layers be even more stable? What are the costs?

14. **Anticipatory Routing's $\Delta t$.** Decoupling routing-index updates from feature updates (using params from step $t - \Delta t$) prevents loss spikes. How big should $\Delta t$ be — and what constrains the upper bound? Is this a temporary fix (a sign that routing-MoE optimization is broken) or a fundamental design insight?

15. **SwiGLU clamping.** Clamping the linear component of SwiGLU to $[-10, 10]$ "eliminates outliers" without compromising performance. Why does this work — what's the population structure of pre-activation values that allows such tight clamping? Is this clamping specific to Muon-trained models, or does it generalize?

---

## DeepSeek-V4: Training and Post-Training

16. **OPD vs. Multi-Domain RL.** V4 *replaces* mixed RL (V3.2's strategy) with **On-Policy Distillation** from independently-trained specialists. What's the conceptual argument for why OPD won? Are there scenarios where joint RL would beat OPD? What does this say about credit assignment in multi-domain RL?

17. **Full-vocabulary KL vs. token-level KL estimator.** V4's OPD computes $D_\mathrm{KL}$ over the full vocab distribution, not just the realized token. Why does this matter for distillation specifically (vs. RLHF, where token-level estimators dominate)? Walk through the variance argument.

18. **FP4-to-FP8 lossless dequantization.** The paper claims that because FP8-E4M3 has 2 more exponent bits than FP4-E2M1, fine-grained FP4 sub-block scales are absorbed into FP8 dynamic range *losslessly* — provided max/min scale ratios within an FP8 block don't exceed a threshold. How does this empirical condition hold up across MoE expert weights vs. attention QK paths? Could a future model violate it?

19. **The $1.6$T-parameter trillion-club bet.** V4-Pro at 1.6T total / 49B activated is an aggressive bet on extreme MoE sparsity. Compared to Llama 3 (405B dense / 405B activated), which approach do you find more compelling? Under what deployment / inference scenarios does each win?

20. **"Think Max" mode.** The injected system prompt for Think Max literally instructs the model to be "thorough", "stress-test logic", and "rigorously decompose." Is this a real capability, or are we training the model to imitate the surface form of careful reasoning? What experiment would distinguish?

---

## DeepSeek-V4: Long Context

21. **The 1M-token claim, in practice.** Figure 9 shows MRCR retrieval is stable to 128K but drops from 0.94 (32K) → 0.59 (1M) for V4-Pro-Max. Is "natively supports 1M" a useful claim if performance halves at the extreme? What concrete deployment scenarios benefit from 200K–500K context that 128K cannot serve?

22. **CSA top-$k$ choice (512 for Flash, 1024 for Pro).** The top-$k$ value sets how many compressed KV blocks each query attends to. How would this scale at 10M tokens? Is there a fundamental information-theoretic argument for what top-$k$ *should* be at a given context length?

23. **Sliding window $n_\mathrm{win} = 128$.** Both CSA and HCA include a small sliding-window branch for local detail. Why exactly 128 (uncompressed)? How sensitive is performance to this choice — and how does it interact with the compression rates $m$ and $m'$?

---

## Cross-Paper / Shared Primitives

24. **mHC and Muon shared between papers.** Both Engram and DeepSeek-V4 use mHC as the residual scaffolding and Muon for the optimizer. What does this say about the *DeepSeek architectural research line*? Are these primitives the foundation for a broader portfolio of upcoming models (V4.5, R2, K3, etc.)?

25. **Two complementary sparsity axes.** Engram attacks the *embedding-lookup* axis; V4 attacks the *attention-sequence* axis. If you combined the two — V4-Pro's attention + Engram's lookup — what would the resulting model look like? Where would the gains compound?

26. **OPD vs. Engram for knowledge.** V4 uses 33T pre-training tokens + OPD distillation for knowledge. Engram instead bakes static knowledge into a dedicated lookup table. These are *very* different approaches to "store knowledge in the model." Which philosophy generalizes better to a new domain?

---

## Connections to Previous Weeks

27. **Mixtral (W6) → DeepSeekMoE → Engram + V4.** Trace the architectural lineage from Mixtral (8 experts top-2, 47B/13B) through DeepSeekMoE to Engram-27B and V4-Pro. What problem does each step solve? Which steps were *necessary* and which were *contingent*?

28. **InstructGPT (W4)'s SFT+RLHF → V4's specialist+OPD.** Both are post-training pipelines, but the structural primitives differ entirely. Map InstructGPT's pipeline to V4's. Which roles do GRMs replace, and which roles do specialists replace?

29. **Mistral 7B SWA (W6) → V4 sliding-window branch.** Both use a fixed local window. But Mistral relies *only* on SWA + multilayer composition; V4 uses SWA *as a supplement* to compressed long-range attention. What changed in the field's understanding of long context that motivated this combination?

30. **DeepSeek-R1 (W8)'s pure RL → V4's specialist + OPD.** R1 showed that pure RL on a strong base produces emergent reasoning. V4 instead trains many specialists then unifies via OPD. Has the field moved away from "let RL do everything"? Why?

---

## Broader Questions

31. **The compute-vs-storage tradeoff at scale.** Engram's argument is that we should store more parameters off-chip (host DRAM, NVMe SSD) and prefetch them into compute. V4's argument is that we should compress the working set so compute-cost-per-token shrinks. Are these competing or complementary visions for the next 1–2 model generations?

32. **Open-source Frontier Gap.** V4-Pro-Max trails Gemini-3.1-Pro by ~17 points on SimpleQA-Verified. What concrete improvements (data curation, post-training, scale) could close that gap? Which is the bottleneck — knowledge depth, retrieval reliability, or instruction-following on knowledge queries?

33. **Cost of the Trillion-Parameter Club.** What are the inference deployment scenarios that justify a 1.6T-parameter model (V4-Pro)? At what scale does the maintenance cost overtake the benefit, and is V4-Flash's 284B / 13B actually the more practical real-world target?

34. **What's the next axis?** Engram introduces *conditional memory* as a new sparsity axis. V4 deeply optimizes attention. What comes next — conditional retrieval (RAG-as-architecture), conditional adaptation (per-token LoRA), conditional decoding strategy? Which will produce the next big jump?
