# Week 10 — Discussion Questions & Suggested Answers
**Paper:** Kimi K2.5: Visual Agentic Intelligence — Technical Report of Kimi K2.5, Kimi Team 2026 (Moonshot AI)

These are suggested answers to guide discussion, not definitive answers. Many of these questions are deliberately open-ended.

---

## Multimodal Pre-training

**1. The early-low-ratio finding.**

The "modality domain shift" argument: when vision is injected late at high ratio, the model has already converged on a stable text-representation manifold. Forcing vision tokens through that manifold disrupts established features — text performance dips before recovering, and the recovery is incomplete because the optimization landscape near the late-fusion injection point is suboptimal for both modalities.

Alternative explanations to interrogate:
- **Total exposure**: under fixed total budget, a 10% ratio means *more* vision tokens *over the course of training* than a 50% ratio applied late. So vision capability could grow simply because the model sees vision tokens more times across training.
- **Optimizer state**: late vision injection requires the optimizer to suddenly handle a different gradient distribution; the inherited momentum from text-only training may bias against new modality features.
- **Learning rate**: by the late stage, LR is decayed; this matters when introducing new task structure.

A clean controlled experiment: hold *total vision exposure* constant (same number of vision tokens), vary only the *temporal placement* of when those tokens are seen. If early-fusion still wins, modality domain shift is real. If they tie, the result is about exposure, not placement.

**2. The 10:90 ratio specifically.**

10% wasn't chosen by sweep — it's the lowest tested ratio. The paper would benefit from a 1% / 5% / 10% / 20% sweep. Theoretical floor: probably around 1-2%, where the gradient signal from vision becomes too weak to reliably update the cross-modal attention paths (signal-to-noise vs. text-only updates becomes too low). Below this floor, the model may "forget" vision capability or never acquire it.

A practical tradeoff: lower vision ratio means more parameter budget goes to language. But if language is the dominant capability and vision is a "supplementary skill" anyway, this trade-off is one the field may have been getting wrong.

**3. MoonViT-3D's parameter sharing.**

The implicit prior: spatial structure dominates visual understanding, and time is a secondary axis that mostly augments rather than transforms it. So image features and video features should live in the same representation space, distinguished only by the additional temporal attention that integrates across frames.

When to break sharing: tasks where temporal reasoning is *qualitatively different* from spatial reasoning — e.g., motion analysis, causal inference about events, action recognition. For these, a separate temporal-only encoder might extract better features than a shared 2D-3D backbone. K2.5's MotionBench result (70.4 — strong) suggests current sharing works for many video tasks, but the limit may be reached on more temporally-complex benchmarks.

**4. YaRN extension to 256K.**

YaRN scales RoPE frequencies by an interpolation factor. With multimodal tokens, the question is: do image / video tokens use the same RoPE frequencies as text tokens? The paper doesn't detail this explicitly, but standard practice is to assign distinct positional encodings per modality (or use no position info for image patches if MoonViT internally encodes spatial structure). If image patches don't use RoPE, YaRN doesn't apply to them — only the text portion of the sequence is affected.

For video, the spatiotemporal volume is treated as a single sequence of tokens, and these tokens *do* receive positional info corresponding to their temporal position in the video (else the model can't distinguish frame ordering). YaRN extension on these would compress effective temporal positions, possibly degrading video understanding at long contexts. The 256K context is *likely* mostly text + sparse video frames, not 256K video frames.

---

## Zero-Vision SFT

**5. Why zero-vision SFT works.**

The connection between writing Python for binarization and visual reasoning: Python code that operates on visual data describes the *operations* of visual reasoning explicitly. When the model is trained to write `cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)`, it's encoding the *semantic* notion of "split image into foreground and background by intensity" into its representations.

When this model is later confronted with a visual reasoning task, it can use either path:
1. *Tool-mediated*: write Python to perform the operation, execute, observe the result.
2. *Direct visual reasoning*: apply the conceptual operation internally without code execution.

Visual RL teaches path (2) by rewarding correct outcomes regardless of the path taken. The zero-vision SFT seeded the *semantic understanding* of what visual operations are; visual RL then trains the model to internalize them.

**6. Comparison to text-vision SFT.**

Three plausible explanations for the paradox:
- **Quality**: text-only SFT data is well-curated and abundant; visual CoT data is hard to write at quality, so existing curated visual CoT may be noisier than its text counterpart.
- **Quantity**: high-quality text SFT exists at much larger scale than visual CoT.
- **Diversity**: text-only SFT covers many domains; visual CoT often clusters around narrow tasks (charts, OCR), missing diverse reasoning styles.

A larger high-quality visual CoT dataset *could* reverse the result, but the cost of curating it competitively is high. Zero-vision SFT is appealing partly because it sidesteps this entire data-curation problem.

**7. Generalization of zero-vision activated capabilities.**

After RL, the model probably mixes both styles. Tool-mediated Python is preferred for tasks where code is naturally expressive (e.g., computing pie chart area percentages — Figure 12), while direct visual reasoning takes over for tasks where code is awkward (e.g., recognizing scene context). The gating is implicit — the model decides per-token whether to write code or reason directly.

What the gating reveals: the model has learned a meta-skill of *modality-of-reasoning selection*, similar to humans deciding "should I do this on paper or in my head?" This is a more interesting skill than either direct reasoning or pure tool-use.

---

## Cross-Modal Transfer

**8. Visual RL improves text benchmarks.**

The "calibration" explanation is partially satisfying: visual tasks involve precise, localized reasoning (count exactly N objects, extract specific OCR text), and this discipline transfers to text where the model now resists vague approximations. But the 1.7-2.2 point gains are modest.

A direct test: train two parallel checkpoints from the same base, one with visual RL added, one without. On structurally similar text tasks (e.g., counting in a passage, structured information extraction), measure improvement specifically. If the calibration story is right, those tasks should see disproportionate gains; if not, the cross-modal gain may just be from broader exposure to RL signals.

**9. The asymmetric transfer claim.**

The paper presents only the visual-to-text direction explicitly. If text-to-visual transfer worked symmetrically, you'd see large vision gains from text-only RL — but the paper instead shows that *zero-vision SFT* is sufficient as a cold start for visual RL, implying text alone doesn't fully transfer. So the transfer is likely asymmetric: visual RL boosts text (via shared structured-extraction skill), but text RL doesn't boost vision (because vision-specific perception ability isn't text-derivable).

If true, this has a striking implication: vision is the more "fundamental" modality from a transfer standpoint, with text being a degenerate (lower-dimensional) projection.

**10. Joint multimodal RL organized by ability, not modality.**

Organizing by ability lets a single skill (e.g., "structured extraction") be optimized through both modalities, accelerating capability acquisition. Organizing by modality would silo the skills and prevent cross-modal transfer.

This is structurally analogous to multi-task learning: tasks that share underlying skill should be co-trained, while tasks that share input format but not skill should be separately optimized. K2.5's choice argues that *abilities* are the right unit of skill, not *modalities*.

---

## Token-Efficient RL: Toggle

**11. Why does pure budget-constrained training fail to generalize?**

When training under a rigid budget, the gradient signal rewards *fitting reasoning into the budget*, not *reasoning effectively*. The model learns truncated reasoning patterns optimized for the budget — and these patterns become the policy. At test time with a larger budget, the model can't suddenly produce the unconstrained reasoning it never learned.

This is a model-side overfitting issue, not a reward signal issue. The reward signal is fine (correctness); the problem is that the policy class explored under budget constraints excludes longer-form reasoning. Toggle solves this by *also* exploring the unconstrained policy class via Phase 1.

**12. Toggle's conditional gate.**

The conditional $\bar{r}(x) < \lambda$ gate prevents *premature compression on hard problems*. Without it: hard problems where the model is still struggling would get budget-suppressed, teaching the model "give up early when stuck" rather than "be concise when capable." This would catastrophically hurt performance on hard benchmarks.

By gating on threshold accuracy, Toggle says: "only push for conciseness on problems you've already mastered." Hard problems remain unconstrained, preserving the model's tail capability. This is reminiscent of curriculum learning, but applied at the per-problem reward level.

**13. The fixed budget percentile $\rho$.**

Fixing prevents a feedback loop where the model's improvement compresses budgets, which forces more compression, which destabilizes training. With a dynamic budget, the moving target makes optimization harder.

Stability concern: a moving budget would interact pathologically with Phase 0 / Phase 1 alternation. In Phase 0, budget would be set by the recent (compressed) policy; in Phase 1, the unconstrained policy might exceed it. Inconsistent reward landscape across phases would prevent convergence.

A fixed budget set early provides a stable target the model can asymptotically approach.

**14. Out-of-domain transfer.**

The gain on GPQA / MMLU-Pro from training on math/coding suggests the model is learning to *compress its existing reasoning patterns*, not just learning to reason concisely about the trained domains. The compression skill (avoid restatements, skip redundant verifications, write more compact intermediate steps) generalizes.

This argues that "reasoning length" has a domain-independent compressible component — the model's verbose-by-default style isn't tied to specific tasks but to a learned global tendency. Toggle reduces this global tendency and the gains transfer.

---

## Agent Swarm

**15. Why Agent Swarm?**

Longer context + more tool budget *can* help, but they don't solve fundamental issues:
- **Sequential cognitive load**: a single context accumulates everything, forcing the agent to maintain state for all parallel research threads at once. This is cognitively (and architecturally) expensive.
- **Noisy context**: irrelevant intermediate results contaminate later reasoning steps in a sequential agent.
- **Deep dependency chains**: some tasks require performing many independent sub-investigations and aggregating; sequential execution forces them to be done in series even when no dependency exists.

Parallelism is *fundamentally* needed when sub-tasks are independent (e.g., research 32 different videos for the Black Myth: Wukong analysis from Figure 11 — no shared state to maintain).

**16. PARL's decoupling.**

The ceiling is real: orchestrator-only optimization can't fix sub-agent flaws. But this is an *intentional* design constraint. Joint training would couple their gradients, destroying credit assignment. PARL chooses *robustness* over *unlimited capability*.

When joint training is worth it: when sub-agent ability is the bottleneck and orchestration is good enough. If sub-agents struggle on a domain-specific task and the orchestrator can't compensate, then joint training (probably at the cost of training stability) might win. But for now, PARL's safety + clean credit assignment seems more valuable than chasing the joint-training ceiling.

**17. The PARL reward structure.**

$r_\mathrm{parallel}$ prevents *serial collapse*: without it, the orchestrator might default to single-agent execution (the easiest local optimum). $r_\mathrm{finish}$ prevents *spurious parallelism*: without it, the orchestrator could spawn many failing sub-agents to inflate parallel-instantiation rewards. The two together act as a Goldilocks scaffold — encouraging parallelism that *also works*.

If you drop $r_\mathrm{finish}$: the orchestrator learns to spawn many sub-agents that may all fail, and $r_\mathrm{parallel}$ rewards this hollow parallelism. Performance degrades because the sub-agents waste compute without producing useful results. This is a reward-hacking failure mode.

The annealing to zero ensures these scaffolds don't *replace* the actual goal ($r_\mathrm{perf}$) at convergence.

**18. Critical Steps as the cost metric.**

Critical Steps explicitly captures wall-clock time in a parallel system. Total step count would penalize parallelism (more agents → more total steps). Wall-clock time is what users care about, so Critical Steps is closer to user value.

A malicious orchestrator could still hack: spawn perfectly balanced but useless sub-agents (each takes 1 step, doing nothing). This would minimize Critical Steps while burning compute. But $r_\mathrm{perf}$ would penalize this — useless sub-agents don't contribute to task success. So combining Critical Steps minimization with $r_\mathrm{perf}$ closes the loop.

**19. Orchestrator's tool budget.**

The asymmetry (15 for BrowseComp, 100 for WideSearch) suggests these tasks have different parallelism profiles built in. BrowseComp tasks usually have a *deep* research path with few branches; WideSearch is *wide* exploration with many parallel searches.

The orchestrator learns these patterns through PARL: rewards from successful tasks shape the orchestrator toward appropriate branching. This may not be entirely emergent — the prompt construction (mention of "wide_search" tasks vs. "BrowseComp"-style queries) likely tunes the orchestrator's prior. PARL fine-tunes within this prior.

**20. Heterogeneous sub-agent specializations.**

The word cloud (Figure 6) — "Biography Researcher," "Verification Specialist," etc. — is most likely *learned* through PARL, not hardcoded. The orchestrator learns to *name* sub-agents in ways that effectively communicate their role to themselves (e.g., a sub-agent named "Verification Specialist" is more likely to perform verification because the system prompt passed to it includes that name).

This is similar to how natural language acts as a form of weak prompt-engineering across the swarm. The stability is task-dependent: the same task likely produces similar specializations across runs (because PARL has learned a general decomposition pattern), but different task types (e.g., BrowseComp vs. video analysis from Figure 11) produce different specialization styles.

---

## Reward Models and GRMs

**21. Multiple alternative GRM rubrics.**

K2.5 likely trains different GRMs with different reward criteria (e.g., one that prioritizes helpfulness, one that prioritizes accuracy, one that prioritizes instruction-following). At training time, these GRMs are rotated or averaged so the policy doesn't overfit to any single rubric. The combination prevents a degenerate solution that maximally satisfies one rubric while hacking the others.

In practice: different GRMs likely come from training on different subsets of preference data with different weights on different criteria. The diversity emerges from human-curated rubric variation rather than algorithmic generation.

**22. Reward hacking surfaces in K2.5.**

The most acute pressure points are likely:
- **Length games**: models inflate response length to seem more thorough — directly addressed by Toggle.
- **Spurious parallelism**: addressed by Critical Steps + $r_\mathrm{finish}$.
- **Instruction-mimicking**: addressed by GRM rubric diversity.

The "instruction-mimicking" failure (model produces text that *describes* doing the work without actually doing it) is particularly insidious in agentic settings — it's hard to catch without rubric-level evaluation. Multi-rubric GRMs help by surfacing this discrepancy.

**23. Visual reward functions.**

The decomposition matters because each task has a *different correctness criterion*. Edit distance for OCR doesn't generalize to grounding (where spatial overlap matters), and IoU doesn't generalize to counting (where absolute count matters). A unified visual reward would either be too coarse (e.g., only "correct/incorrect") or require complex rubric-based scoring.

The modal multi-reward design is practical and effective. Future work might develop a learned unified visual reward via a single GRM trained on diverse visual outcomes — but this would require collecting human preference data for visual tasks, which is more expensive than text.

---

## Architecture and Infrastructure

**24. Decoupled Encoder Process (DEP).**

The trade-off in the three-stage process: stage 1 produces vision encoder outputs; stage 2 trains the LLM backbone (which uses these outputs); stage 3 backpropagates *back through the vision encoder*. Re-computing the vision encoder forward in stage 3 (instead of caching activations from stage 1) saves activation memory at the cost of one extra forward.

This is a memory-time trade-off. When activation memory is the bottleneck (e.g., at large image batches), re-computation pays off. When wall-clock time is the bottleneck, caching wins.

**25. MoonViT-3D's 4-frame compression.**

4 frames at 30 FPS = 0.13 seconds — enough to capture smooth motion (1 frame per 33ms) but not high-speed events (faster than 8 Hz visual events). Below 4 frames, you lose useful temporal context; above 4 frames, intra-volume motion gets averaged into the spatial features, blurring temporal info.

For higher-resolution video or denser temporal events (sports, fast-motion gaming), the 4-frame design starts losing fidelity. The upper bound on practical video length is determined by total token budget — at 256K context tokens with ~1000 tokens per spatiotemporal volume, you can process ~250 spatiotemporal volumes = ~1000 frames = ~33 seconds at 30 FPS. For longer videos, K2.5 likely subsamples frames.

**26. Token-in-Token-out paradigm.**

Train-inference mismatch correction: during RL training, the same prompts are processed by both the inference engine (for rollouts) and the training engine (for gradient computation). If these engines compute logits slightly differently (due to different kernels, attention implementations, or numerical precision), the training engine may apply gradients based on a different distribution than the one used for sampling. This is the classic *train-inference mismatch*.

The fix: at sampling time, record the actual log probability of each token chosen. At training time, use this recorded log prob (not the freshly-computed one) for the policy ratio. This makes the training mathematically consistent with the sampling process.

This is similar to standard PPO's importance sampling, but more aggressive — instead of trusting the inference engine to be approximately consistent, K2.5 explicitly verifies and corrects.

---

## Evaluation and Frontier Gap

**27. Knowledge gap vs. proprietary models.**

The 35-point SimpleQA-Verified gap to Gemini-3.1-Pro is huge. Gemini's advantages:
- **Web-scale fact verification training data**: Google has unique access to high-quality factual data with verified provenance.
- **Live retrieval at inference**: Gemini 3 Pro probably blends internal knowledge with retrieval (not pure parametric knowledge).
- **Targeted factuality post-training**: Gemini likely has specialized RLHF on factual responses.

Open-source K2.5 could close part of this with:
- Better fact-curation in pre-training (very expensive without proprietary data sources).
- Inference-time RAG integration (retrofittable, but doesn't help on closed-book benchmarks).
- Targeted post-training for factual responses (achievable but requires labeled factuality data).

The gap is most likely permanent on closed-book SimpleQA without proprietary data access.

**28. Coding parity, not lead.**

SWE-Bench Verified is a curated, controlled benchmark where Opus 4.5's strengths in long-context code understanding shine. LiveCodeBench v6 introduces fresh problems closer to test-time, where K2.5's recent training data and possibly different problem distribution helps.

Which is more meaningful: LiveCodeBench v6 better tests *generalization* (out-of-distribution problems), while SWE-Bench Verified tests *real-world relevance* (real GitHub issues). Both are useful, but for production deployment, SWE-Bench Verified is the more concrete signal.

**29. Visual benchmarks dominance.**

Likely a combination:
- **Joint multimodal RL** trains vision capabilities alongside text capabilities, getting cross-modal benefits.
- **MoonViT-3D** is specifically designed for native-resolution image processing with strong 2D-3D parameter sharing.
- **Zero-vision SFT** gives the model fluent code-mediated visual reasoning.
- **Outcome-based visual RL** trains the model on actually correct outputs, not just reasonable-looking ones.

The combination is what beats prior approaches that excel at one of these dimensions. K2.5's investment in multimodal post-training is what closed the gap to dedicated VLMs.

**30. Computer use breakthrough.**

OSWorld-Verified requires:
- **Long-horizon planning**: GUI tasks span many steps.
- **Visual understanding of arbitrary interfaces**: native-resolution visual processing matters.
- **Tool-use without external help**: pure GUI actions, no specialized tools.
- **Robust state tracking**: re-grounding the screen as it changes.

K2.5's strengths align with these: agentic RL training, MoonViT-3D, zero-vision SFT-bootstrapped tool use, and Toggle's reasoning efficiency. Other open-source models (Qwen3-VL) lack the agentic post-training.

GPT-5.2's score of 8.6 is striking — likely an evaluation artifact (10% no-output failure rate counted as wrong) rather than a true capability gap.

**31. Agent Swarm's gain magnitude.**

Likely *additive* with reasoning effort, not trade-off — Agent Swarm parallelizes external tool use, while reasoning effort intensifies internal reasoning. They operate at different levels of the agent stack.

The upper bound: as long as the orchestrator can identify parallelizable structure in the task, Agent Swarm gains compound. For purely sequential tasks (where each step strictly depends on the previous), Agent Swarm provides little benefit. The fraction of parallelizable real-world tasks is large but not 100%, so the practical ceiling on Agent Swarm gains is real but unknown.

---

## Connections to Previous Weeks

**32. DeepSeek-R1 (W8)'s emergent CoT → K2.5's Agent Swarm.**

R1's CoT is *intra-agent* reasoning — a single sequence of tokens that includes thought-trace markers. K2.5's Agent Swarm is *inter-agent* reasoning — orchestrator + parallel sub-agents communicating through tool calls.

Which is "deeper": neither — they're *complementary*. CoT is depth-of-reasoning per-problem; Agent Swarm is breadth-of-reasoning across sub-problems. A truly capable agent needs both: deep CoT for the orchestrator's planning and the sub-agents' analysis, plus the parallel decomposition that Agent Swarm provides.

K2.5 actually combines these: each agent (orchestrator and sub-agents) does CoT, and Agent Swarm coordinates them.

**33. DeepSeek-V4 (W9)'s OPD vs. K2.5's RL pipeline.**

V4's OPD: distill from many specialists to one student. Avoids credit assignment in joint RL.

K2.5's joint RL: optimize all domains simultaneously. Aware of credit assignment but bets on the diversity of GRM rubrics + token-level clipping to keep training stable.

Strengths/weaknesses:
- V4 is more conservative: stable training but capped at specialist quality. K2.5 is more aggressive: potentially higher ceiling but harder to train.
- V4 separates concerns: specialists train independently then merge. K2.5 entangles concerns: joint multi-domain RL allows cross-domain capability fertilization.

For K2.5's *agentic* setting (where capabilities span tool use, vision, reasoning, instruction-following), joint optimization may be necessary — these capabilities must be tightly integrated for long-horizon tasks. For a more compartmentalized model, V4's approach is cleaner.

**34. InstructGPT (W4)'s SFT+RLHF → K2.5's SFT+joint RL.**

| Component | InstructGPT (W4) | K2.5 |
|-----------|------------------|------|
| SFT | Demonstration data | Multi-source: high-quality from K2/K2-Thinking, expert models, zero-vision SFT |
| Reward signal | Trained scalar reward model on human preferences | Rule-based (verifiable tasks) + GRM (open-ended) + visual rewards |
| RL algo | PPO | K1.5-style with token-level clipping |
| Multi-domain | Single domain | Many domains (knowledge, reasoning, coding, agentic) jointly |
| Modality | Text-only | Multimodal (text + vision) |

What's the same: SFT → RL pipeline, learned reward signal, importance-sampling-based policy update.

What's fundamentally different: K2.5's multi-domain joint RL with multiple reward types (rule-based + GRM + task-specific visual) — InstructGPT's single reward model can't capture this heterogeneity. K2.5's token-level clipping (vs. PPO's standard clip) handles long-horizon training. K2.5's GRMs are *generative* models, not scalar predictors.

**35. Mixtral (W6) → Kimi K2 1.04T MoE.**

Largest gap in understanding: the *training stability* of K2's 384-expert system. Mixtral with 8 experts is essentially a small MoE; load imbalance and routing instability are easily handled with auxiliary loss. K2's 384 experts at trillion-scale require:
- DeepSeekMoE-style fine-grained experts + shared experts
- Auxiliary-loss-free balancing (vs. Mixtral's auxiliary loss)
- MuonClip optimizer with QK-Clip (Mixtral used Adam)
- Sophisticated infrastructure for EP and DP coordination

The gap is probably in the *degree of engineering investment*. Mixtral worked because the design space at 8 experts is small. K2 works at 384 experts because every component (router, expert sizing, optimizer, infra) was specifically engineered for this scale. Without that engineering, naive scaling of Mixtral to 384 experts would fail.

---

## Broader Questions

**36. Visual agentic intelligence as a milestone.**

It's a meaningful capability tier, but not yet "general-purpose." What's missing:
- **Embodied robotics**: K2.5 understands video and computer interfaces, not physical environments.
- **Persistent memory**: each conversation starts fresh; no learned user preferences or ongoing project state.
- **Real-time adaptation**: K2.5 is post-trained, not continually learning.
- **Genuine multimodal generation**: K2.5 generates text + tool calls; not images, audio, or video natively.

Adding embodied robotics and persistent memory would make a true general-purpose visual agent — but each is a research program in itself.

**37. Open-source Agent Swarm equivalents.**

Could *partially* be retrofitted. The orchestrator's tool-calling behavior is a function of the base model's instruction-following capability — Llama 3 / Qwen could play orchestrator with appropriate prompt scaffolding. But the *learned* behavior (knowing when to parallelize, what specializations to instantiate, how to balance the load) is entrenched in K2.5's RL training.

A retrofit would get crude functionality (instantiate sub-agents per a manual script) but miss the adaptive aspects (dynamic decomposition based on task structure). For PARL-quality orchestration, you'd need to apply the PARL framework to a different base model — a non-trivial research effort, not a deployment move.

**38. Cost of Agent Swarm at deployment.**

Wall-clock savings of 4.5× don't equal compute savings. Total compute = orchestrator compute + sum of sub-agent computes. If you spawn 14 sub-agents (Figure 4 mean), the total compute can be 14× a single-agent baseline.

What you save: latency (which is what users perceive), and possibly cost-per-result-quality (if Agent Swarm achieves higher accuracy per dollar). What you don't save: raw FLOPs.

For latency-sensitive deployments (interactive agents), Agent Swarm is a clear win. For batch / cost-sensitive deployments (offline agents), it may not be — unless the parallelism enables shorter total wall-clock that lets the cluster handle more requests.

**39. Evaluation contamination.**

Real concern, especially for AIME/HMMT 2025 (Feb) and Codeforces 2025. The Kimi K2 base was trained through 2025; problem-solving content from these benchmarks could have leaked into pre-training. Even if the exact problems were excluded, *similar* problems with similar solution patterns would inflate scores.

Counter-evidence: K2.5 doesn't dominate AIME 2025 (96.1 vs. GPT-5.2's 100, Gemini's 95.0) — if contamination were severe, you'd expect outright wins. The fact that proprietary models are competitive suggests contamination effects are similar across the board.

But: comparing K2.5 to non-thinking baselines (V3.2 at 93.1) shows K2.5 has a real advantage. Some of this is genuine capability; some may be data freshness. Reliable disentanglement requires *held-out* benchmarks created after the training cutoff, which the field is still developing.

**40. The "frontier gap" estimate.**

V4-Pro lags by 3-6 months. K2.5 reaches comparable performance now. The trajectory:
- Open-source labs (DeepSeek, Moonshot, Alibaba) are converging on similar techniques (MoE, MuonClip, GRPO/PPO variants, multi-domain RL).
- Each generation closes the gap by ~1-2 months as techniques mature.
- Proprietary labs (OpenAI, Anthropic, Google) have access to richer data sources and more compute, but research velocity in open-source has caught up significantly.

Realistic forecast: open-source frontier reaches parity or near-parity within 12-18 months on most tasks, with continued lag on knowledge benchmarks (where data access matters most). The structural gap is *data*, not architecture or training. Whether proprietary labs can accelerate to maintain a gap depends on their willingness to invest in *fundamentally* new approaches (not just scale) — which is currently uncertain.
