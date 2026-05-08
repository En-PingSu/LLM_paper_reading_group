# Week 10 — Discussion Questions
**Paper:** Kimi K2.5: Visual Agentic Intelligence — Technical Report of Kimi K2.5, Kimi Team 2026 (Moonshot AI)

---

## Multimodal Pre-training

1. **The early-low-ratio finding.** Table 1 shows that early fusion at a 10:90 vision:text ratio outperforms late fusion at 50:50 — across both vision *and* text benchmarks. Walk through the authors' "modality domain shift" argument. What alternative explanations could there be (e.g., learning rate effects, optimizer effects, total exposure of vision vs. ratio)? How would you design a controlled experiment to disentangle these?

2. **The 10:90 ratio specifically.** Why 10%? The paper doesn't sweep below 10% (e.g., 5%, 1%, 0.1%). At what point would vision contribution become so diluted that text dominates entirely? Is there a theoretical floor below which the model cannot acquire visual capability?

3. **MoonViT-3D's parameter sharing.** The 3D extension shares parameters between image and video paths, with only an additional temporal attention component for video. What's the inductive prior here — what does this imply about how images and videos are *meant* to relate in representation space? When would you want to break the parameter sharing?

4. **YaRN extension to 256K.** The third pre-training stage uses YaRN to extend context length from 32K → 256K. YaRN-style extensions are typically applied to text-only RoPE. How does YaRN interact with multimodal tokens (e.g., the spatiotemporal video volumes)? Does the same interpolation factor apply to all token types?

---

## Zero-Vision SFT

5. **Why zero-vision SFT works.** The argument: text-only SFT data with image manipulations as IPython operations (binarization, counting, OCR) suffices to bootstrap visual capability for downstream RL. Walk through *why* this transfers — what's the connection between writing Python code to do binarization and the model "learning visual reasoning"?

6. **Comparison to text-vision SFT.** The paper claims zero-vision SFT outperforms text-vision SFT (with curated visual CoT data) on downstream visual/agentic tasks. This is counterintuitive. Is the issue *quality* of curated visual CoT, *quantity*, or *diversity*? Could a much larger visual CoT dataset reverse the result?

7. **Generalization of zero-vision activated capabilities.** Once visual RL is bootstrapped, does the model retain the IPython-mediated style? Or does the post-RL behavior shift entirely toward natural visual reasoning? What does the gating mechanism reveal about how the model decides between "use Python" vs. "reason directly"?

---

## Cross-Modal Transfer

8. **Visual RL improves text benchmarks (Table 2).** MMLU-Pro +1.7, GPQA-Diamond +2.1, LongBench v2 +2.2. The paper attributes this to "calibration in areas requiring structured information extraction." Is this a satisfying explanation? What experiment would directly test the proposed mechanism?

9. **The asymmetric transfer claim.** The paper implies cross-modal transfer is bidirectional (visual RL → text gains, presumably text RL → vision gains). Is there evidence in the paper that *both* directions transfer equally? If asymmetry exists, what does that say about the underlying capability structure?

10. **Joint multimodal RL organized by ability, not modality.** Domains are knowledge, reasoning, coding, agentic — each ingesting both text and multimodal queries. Why is this the right organization? What would happen if domains were organized by modality (text-only, image-only, video-only) instead?

---

## Token-Efficient RL: Toggle

11. **Why does pure budget-constrained training fail to generalize to higher compute scales?** The paper observes "length-overfitting" — models trained under rigid budgets default to truncated reasoning even when given more tokens. Why? Is this a model-side overfitting issue, or a problem with the reward signal?

12. **Toggle's conditional gate.** Phase 0's reward is gated by $\mathbb{I}[\bar{r}(x) < \lambda \;\text{or}\; |y| \leq \mathrm{budget}(x)]$ — only enforce the budget when the model is *already* solving the problem at threshold accuracy. Walk through this design. What goes wrong if you don't gate, i.e., always enforce budget?

13. **The fixed budget percentile $\rho$.** $\mathrm{budget}(x)$ is the $\rho$-th percentile of correct-response lengths, computed once at training start. Why fix it? Wouldn't a dynamic budget (re-computed periodically) be better as the model improves? What stability concerns motivate fixing it?

14. **Out-of-domain transfer.** Toggle shows that training exclusively on math/coding still produces token reductions on GPQA and MMLU-Pro with marginal accuracy degradation. What does this suggest about the source of token efficiency — is the model learning to reason concisely, or learning to *compress* its existing reasoning?

---

## Agent Swarm

15. **Why Agent Swarm?** Sequential agentic execution exhausts reasoning depth, tool budget, and context. But couldn't simply "longer context + more tool budget" address this? Walk through the specific scenarios where parallelism is *fundamentally* needed, not just convenient.

16. **PARL's decoupling.** Sub-agents are frozen at intermediate policy checkpoints; only the orchestrator is RL-trained. This avoids "credit assignment ambiguity." But this also caps the system's ceiling at what the frozen sub-agents can do. When would joint orchestrator + sub-agent training be worth the complexity?

17. **The PARL reward structure.** $r_\mathrm{parallel}$ + $r_\mathrm{finish}$ + $r_\mathrm{perf}$, with the first two annealed to zero. Why exactly two scaffolding rewards (instantiation + finish), not one? What failure mode does each prevent? What goes wrong if you drop $r_\mathrm{finish}$?

18. **Critical Steps as the cost metric.** Defined as $\sum_t (S_\mathrm{main}^{(t)} + \max_i S_{\mathrm{sub},i}^{(t)})$. Why is this the right cost? Could a malicious orchestrator still reward-hack by, e.g., creating sub-agents that are deliberately balanced but wasteful?

19. **Orchestrator's tool budget (15 steps for BrowseComp, 100 for WideSearch).** The asymmetry suggests BrowseComp is treated as "deep research, few branches" while WideSearch is "wide exploration, many branches." How does the orchestrator learn to choose the right branching factor for each task? Is there an explicit signal, or does it emerge from PARL?

20. **Heterogeneous sub-agent specializations (Figure 6).** The word cloud shows organically emergent specializations like "Biography Researcher," "Verification Specialist," "Cross Reference Analyst." Was the diversity learned, or instructed via prompt construction? How stable / interpretable are these emergent specializations across different tasks?

---

## Reward Models and GRMs

21. **Multiple alternative GRM rubrics.** K2.5 employs "multiple alternative GRM rubrics tailored to different task contexts" to mitigate reward hacking. Walk through this design. How are multiple rubrics trained and combined at training time?

22. **Reward hacking surfaces in K2.5.** What specific reward-hacking failures has the team likely encountered to motivate the multi-rubric approach? Spurious parallelism (Agent Swarm), length games (Toggle), instruction-mimicking-without-execution (general RLHF) — which is the most acute pressure point?

23. **Visual reward functions.** Visual RL uses task-specific reward functions: F1 with soft-matching for grounding, IoU for segmentation, edit distance for OCR, absolute difference for counting. Are these decomposable into a unified visual reward, or is the modal multi-reward design necessary?

---

## Architecture and Infrastructure

24. **Decoupled Encoder Process (DEP).** The three-stage process — Balanced Vision Forward → Backbone Training → Vision Recomputation & Backward — re-computes the vision encoder forward in stage 3. What's the trade-off here, and when would you prefer storing intermediate vision activations instead?

25. **MoonViT-3D's 4-frame compression.** Consecutive 4 frames are treated as a single spatiotemporal volume. Why 4 specifically? At what frame rate / resolution does this stop being information-preserving? What's the upper bound on practical video length given this design?

26. **Token-in-Token-out paradigm.** The RL framework records log probabilities for all output tokens to enable train-inference mismatch correction. Walk through what mismatch correction does. How does this differ from standard PPO importance sampling?

---

## Evaluation and Frontier Gap

27. **Knowledge gap vs. proprietary models.** Kimi K2.5 trails Gemini-3.1-Pro by 35 points on SimpleQA-Verified (37 vs. 72). This is a much larger gap than on most other benchmarks. What is Gemini doing that K2.5 isn't? Is this a base-model issue, a post-training issue, or a tool-integration issue?

28. **Coding parity, not lead.** K2.5 achieves SWE-Bench Verified 76.8 vs. Opus 4.5's 80.9. On LiveCodeBench v6 (live, harder distribution shift), K2.5 leads at 85.0 vs. Opus's 82.2. Walk through why the relative ranking differs across benchmarks. Which is the more meaningful capability test?

29. **Visual benchmarks dominance.** K2.5 leads on MathVision (84.2), OmniDocBench (88.8), OCRBench (92.3), InfoVQA (92.6). What's the source of this consistent advantage — the joint multimodal RL training, the MoonViT-3D architecture, or something else?

30. **Computer use breakthrough.** OSWorld-Verified 63.3 substantially outperforms open-source baselines (Qwen3-VL-235B's 38.1, GPT-5.2's 8.6). What enables this? What do real GUI agents need that other multimodal benchmarks don't?

31. **Agent Swarm's gain magnitude.** BrowseComp 60.6 → 78.4 (+17.8) is a striking gain. Are these gains additive with reasoning effort (Think Max), or do they trade off? What's the upper bound of capability extracted from a single base model via better orchestration?

---

## Connections to Previous Weeks

32. **DeepSeek-R1 (W8)'s emergent CoT → K2.5's Agent Swarm.** R1 showed emergence of reasoning *within* a single agent's CoT. K2.5's Agent Swarm distributes reasoning *across* multiple agents. What's the relationship — and which is "deeper reasoning"?

33. **DeepSeek-V4 (W9)'s OPD vs. K2.5's RL pipeline.** V4 distills specialists; K2.5 jointly trains via RL. Both target a single unified model from heterogeneous training signals. Compare strengths and weaknesses.

34. **InstructGPT (W4)'s SFT+RLHF → K2.5's SFT+joint RL.** Both are post-training pipelines for a base model. Map K2.5's components to InstructGPT's. What's the same, and what's fundamentally different?

35. **Mixtral (W6) → Kimi K2 1.04T MoE.** Mixtral was 47B/13B with 8 experts top-2. Kimi K2 is 1.04T/32B with 384 experts (8 active, sparsity 48). What's the largest gap in your understanding between these two, in terms of why one design works at this scale and the other doesn't?

---

## Broader Questions

36. **Visual agentic intelligence as a milestone.** Is "visual agentic" a new capability tier, or just the convergence of trends (multimodal + agentic)? What's missing from K2.5 to make it a true general-purpose visual agent — embodied robotics, persistent memory, real-time adaptation?

37. **Open-source Agent Swarm equivalents.** Could an Agent Swarm framework be retrofitted onto an existing open-source model (e.g., Llama 3, Qwen) without re-training? Or is the orchestrator's behavior entrenched in the K2.5 base?

38. **Cost of Agent Swarm at deployment.** Agent Swarm reduces wall-clock by 4.5× but instantiates many concurrent sub-agents. What does the *total cost* (compute, memory, context) look like for a complex BrowseComp task? Is the savings in latency, or also in cost?

39. **Evaluation contamination.** AIME 2025, HMMT 2025 (Feb), Codeforces 2025 — these are recent benchmarks. To what degree could K2.5's training data have inadvertently included problem-solving content from the same period? How much of the gap from Gemini-3.1-Pro is due to inadvertent contamination vs. genuine capability?

40. **The "frontier gap" estimate.** If V4-Pro lags frontier by ~3-6 months and K2.5 reaches comparable performance now, the open-source frontier is moving fast. What's the realistic path forward — does this trajectory continue, or do proprietary labs accelerate via internal compute scaling?
