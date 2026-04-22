# Week 8 — Discussion Questions & Suggested Answers
**Paper:** DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning, DeepSeek-AI, January 2025

These are suggested answers to guide discussion, not definitive ones. Many of these questions are deliberately open-ended.

---

## The "Pure RL" Thesis

**1. Does R1-Zero actually prove reasoning is incentivized rather than taught?**

This is the paper's strongest claim and also its most contested. The honest answer is: *R1-Zero proves reasoning can be amplified via RL from a strong base, but not that RL creates reasoning from nothing*. Appendix A.1 explicitly acknowledges DeepSeek-V3-Base was "exposed to a significant volume of reasoning trace data" during pretraining. The paper's own Appendix G.1 reports that *smaller* base models (7B dense, 16B MoE) failed to RL-train reasoning — response length grew without accuracy following. This is important: if RL truly *created* reasoning, small models should work too (just with more steps). The fact that they don't suggests RL is surfacing capability latent in the base. A falsifying experiment would be: train a large base model on carefully filtered data with no reasoning content (no math solutions, no competitive programming), then apply R1-Zero's recipe. If reasoning still emerges, the strong claim holds; if not, RL is an amplifier, not a creator.

**2. The aha moment — genuine emergence or selection bias?**

Table 2 is selected, by the authors' own admission ("an interesting 'aha moment'... This is also an aha moment for us"). The frequency is quantified in Appendix C.2 / Figure 9: reflective words rise 5–7× over training, and "wait" spikes near step 8,000. That's weak evidence for an emergent *capability* but strong evidence for an emergent *verbal register*. What's missing: a correlation between "presence of reflective language" and "probability of correctness", broken out by problem difficulty. Without that, we can't distinguish "the model actually re-examines its work" from "the model has learned to write re-examination language because that register is correlated with correct answers in its training distribution." A discussion-worthy experiment: ablate away the "wait"-containing traces at inference and see if accuracy drops.

**3. Why does response length grow?**

Several mechanisms likely contribute: (a) longer outputs get more chances to hit the correct answer — a model that writes out intermediate computations is less likely to fumble arithmetic; (b) longer reasoning reduces the variance of its own predictions, so the RL gradient is on firmer ground; (c) GRPO's advantage normalization means *relatively* longer outputs that happen to be correct get a positive signal, and the policy interpolates toward that regime. Could it be a failure mode? Yes — the paper's limitation section flags "overthinking" on simple problems. The clearest sign this is load-bearing rather than cosmetic is Figure 18: longer outputs correlate with harder problems. If length were just runaway token-padding, it would be roughly flat across difficulty.

**4. What does "no SFT" really mean?**

It means no instruction tuning, no reasoning demonstrations, no chat-style fine-tuning — just the pretrained next-token predictor. But "pretrained" is doing a lot of work here: V3-Base saw math textbooks, olympiad solutions, and Stack Overflow during pretraining, and the paper describes a "pretraining cooldown phase" that excluded synthetic data. So R1-Zero starts from a model that can already *continue* a reasoning trace if prompted — it just isn't aligned to format its output helpfully. The paper's framing ("obviating the need for human-labeled reasoning trajectories") is technically accurate about post-training but elides the pretraining data story. For discussion: does the distinction matter? If you're building a reasoning model today and you already have a strong base, the R1-Zero recipe is a valid starting point regardless of whether the thesis is "creation" or "amplification".

---

## GRPO and the RL Algorithm

**5. Why does dropping the value model work?**

The core issue with a value model on long CoT is that partial-trajectory value is ill-defined when the model can reverse itself. At token 5,000 the model might be about to write a correct answer; at token 5,001 it might write "wait, let me reconsider" and flip. A value function trained to predict the eventual reward from the first 5,000 tokens would be highly inconsistent. GRPO sidesteps this entirely: it only ever compares *complete* trajectories. The tradeoff: you need $G$ samples per prompt to get a usable advantage signal, and the variance is $O(1/\sqrt{G})$. PPO with a well-trained value model can give you per-token variance reduction for free. GRPO would fail when: (a) the space of good outputs is very narrow so random samples rarely succeed (high reward sparsity), or (b) compute is constrained so you can't afford $G{\geq}8$. On reasoning with $G{=}16$ and correctness rates of 20–80%, the statistics are fine.

**6. Is GRPO actually novel?**

GRPO is from Shao et al. 2024 (DeepSeekMath). This paper's contribution is applying it to general reasoning at scale and reporting the reward hacking / clip-ratio lessons. PPO with a value model could *probably* reach similar conclusions with better engineering (e.g., periodic value model retraining, larger value models) but at higher cost. The scientific claim about RL-incentivized reasoning doesn't hinge on GRPO specifically — it hinges on "outcome-based RL works on long CoT". That said, GRPO's memory savings may have been *practically* necessary given DeepSeek's hardware constraints (64×8 H800s).

**7. Group size as a hyperparameter.**

$G{=}2$ degenerates to pairwise preference-style learning (like DPO), and the advantage magnitude becomes binary (±1 after normalization). $G{=}64$ gives finer-grained ranking but costs 4× the rollout compute. $G{=}16$ is likely a sweet spot where: (a) the batch has enough variance to produce usable mean/std, (b) some outputs succeed and some fail on most prompts (reward isn't degenerate), (c) rollout cost is tolerable. The paper doesn't ablate $G$, which is a conspicuous omission — it would be a great follow-up experiment.

**8. The clip ratio of 10.**

Standard PPO uses $\varepsilon{=}0.2$, meaning $\pi_\theta(o|q) / \pi_{\theta_{old}}(o|q)$ is clipped to $[0.8, 1.2]$. With $\varepsilon{=}10$, the clip is $[-9, 11]$ — effectively no clipping for 99% of tokens. This reveals two things: (a) the importance ratios in long-CoT training routinely blow up beyond standard PPO's assumed range, probably because each rollout's inner epoch updates push the policy far from the old policy; (b) without effectively unclipped updates, many gradient contributions get zeroed. It's a sign the PPO-style objective isn't quite right for this regime — or that the "old policy" refresh should be more frequent to keep ratios tight. The authors acknowledge this is load-bearing and warn against low clip ratios.

**9. Why is KL divergence so low?**

$\beta{=}0.001$ is indeed ~100× weaker than InstructGPT. The reason it's tolerable: the reward is not a learned function that can be hacked into producing nonsense — it's a compiler or a math-checker. If the policy drifts far from the reference, the only way to keep getting high reward is to actually solve the problem. In RLHF with a neural reward model, the reference is a sanity check that prevents the policy from gaming the RM's blind spots. In rule-based-reward RL, there are no blind spots to game, so you can let the policy drift. The risk: you lose general fluency / coherence on tasks the rule-based reward doesn't cover. R1-Zero's readability problems are exactly this — the language model has drifted from the base, and unconstrained rule-based RL let it.

---

## Reward Design

**10. Why reject neural reward models for reasoning?**

Two compounding issues: (a) neural RMs have blind spots — sentence-level features that correlate with "looks like a good answer" without being it, and RL discovers these quickly; (b) RMs trained on off-policy data go stale as the policy evolves, and retraining is expensive. The authors' honest admission: this approach works *because* reasoning has verifiable ground truth. It doesn't generalize to creative writing, strategic advice, or design tasks. The implicit claim is that the *reasoning core* of RL should stay rule-based, and neural RMs should only appear in the final "polish" stage where they're kept on a short leash (Stage 3 activates preference RMs only in the final 400 steps). Does this generalize? Probably to any domain with programmable verifiers — theorem proving, certain scientific simulations, structured data extraction. Not to open-ended domains.

**11. Accuracy + format, equally weighted.**

Equal weighting is pragmatic: both rewards are in $\{0,1\}$ so sum is in $\{0,1,2\}$. Up-weighting accuracy would reduce the policy's incentive to maintain format, risking answers the evaluator can't parse. The failure mode the authors likely worried about: a policy that gets the right answer but loses the `<answer>` tags, making automatic evaluation impossible. By making format a first-class reward, the template is always produced. Whether format-first → correctness-second is an issue: in practice format is trivial to learn (probably converges within the first few hundred steps), so after early training the format reward is near-saturated and accuracy dominates the gradient.

**12. The language consistency reward costs accuracy.**

This is honest and notable. The 1–2 point drop on math benchmarks suggests the model's "most efficient" reasoning language is sometimes bilingual (use Chinese for numerical work, English for chain-of-thought, or vice versa). Forcing monolingual CoT handicaps the reasoning. A principled vs pragmatic read: the authors chose it as a pragmatic UX fix, but it's flagging a deeper tension — the model's internal representation may genuinely be more efficient when language isn't constrained. This connects to observations in interpretability work that multilingual models operate on abstract "interlingua" representations; forcing a surface language may be squeezing through a suboptimal projection.

**13. Format reward as structural prior.**

The `<think>/<answer>` template almost certainly provides scaffolding beyond what the reward alone captures. It gives the model a syntactic context where "thinking out loud" is expected, reducing the cost of exploration. Without the template, the model would have to learn *when* to start reasoning and *when* to commit — the format makes that free. An interesting ablation the paper doesn't run: try pure accuracy reward, no format reward, no template — does reasoning still emerge? We'd bet it does, but much more slowly, and the output would be messier to parse.

---

## The Multi-Stage Pipeline

**14. Why does cold-start SFT hurt AIME?**

Cold-start SFT teaches a *style* (first-person, clean formatting) but only has thousands of examples. That's not enough to re-teach reasoning strategies, so it partially overwrites R1-Zero's strategies while enforcing format. The recovery in Dev2 (Reasoning RL restores most of the loss) suggests the reasoning strategies were latent in Dev1 — cold-start didn't destroy them, it just suppressed them temporarily. This is consistent with a standard SFT-then-RL pattern: SFT forms the distribution, RL sharpens it within that distribution. The cold start pays for itself by the time you reach R1 because the readable format is what makes subsequent stages (rejection sampling, general RL) tractable.

**15. Why not R1-Zero → cold-start → RL in a single pipeline?**

The paper doesn't explicitly say, but likely reasons: (a) R1-Zero's weights have drifted far from V3-Base, possibly in ways that make the cold-start SFT data less relevant; (b) starting cold-start from V3-Base means the subsequent reasoning-RL stage sees a model distribution closer to the one that worked for R1-Zero; (c) cleanliness for publication — it's easier to argue each pipeline stage is valuable when they cleanly stack. An alternative interpretation: they tried both and didn't share the ablation. This is a place where a direct experiment would be informative.

**16. Four stages — is this the minimum?**

Plausibly each stage has a distinct role:
  - Cold-start SFT: enforces style and format.
  - Reasoning RL: rebuilds reasoning within the SFT'd style.
  - Rejection SFT: expands competence to general tasks with self-distilled data.
  - General RL: aligns with user preferences.

You could probably collapse Stages 2 and 3 into a single RL stage with mixed rewards, but you'd risk reward hacking on the preference RM. You could skip Stage 0 (cold-start) and take R1-Zero's readability hit as a feature. The load-bearing stages are plausibly 1 (reasoning RL) and 2 (rejection SFT for generalization). Cold start and general RL are polish.

**17. Using DeepSeek-V3 as judge.**

V3 is weaker at reasoning than R1-Dev2 but comparable at *detecting* correctness when given the ground truth answer. The judge's task isn't "solve the problem" — it's "does this answer match the reference?" That's a much easier task and V3 is reliable at it. The paper also describes the LLM-as-judge being used with a structured prompt (Listing 4) that frames the task as classification, not generation. A genuinely weaker judge could still fail if the model exploits ambiguity ("is 0.5 the same as 1/2?"), but for well-formatted final answers this is mostly fine.

**18. General RL only moves the preference benchmarks.**

Stage 3 uses two preference RMs (helpfulness + safety) and only trains for 1,700 steps. The reasoning benchmarks are already near-ceiling by Dev3, so there's little room to move — but reasoning could also be silently degrading and being masked by benchmark saturation. The paper doesn't break out "reasoning under adversarial preference pressure" as a metric. A skeptic would ask: what happens on *novel* reasoning problems after Stage 3? Does the model start preferring short, "helpful"-looking responses over deep reasoning? Worth discussing whether the preference-RM final stage could be slowly eroding reasoning robustness in ways the current benchmarks don't catch.

---

## Distillation vs RL

**19. Distill-32B beats Qwen-32B-Zero by 25 AIME points.**

This is one of the paper's most consequential findings. The interpretation is: small models can't discover good reasoning strategies on their own via RL, but they can *imitate* strategies a larger model has already discovered. This aligns with Hinton-style distillation insights: the teacher's soft labels (full reasoning traces, in this case) pack information about the structure of the solution space that the student couldn't have found from scratch. Implication for practice: if you have a strong teacher, distill; don't burn compute training small models with RL.

**20. When should you run RL on top of a distilled model?**

The paper punts on this — "we leave this to the research community." A plausible hypothesis: distilled + RL would recover roughly R1's AIME ceiling (since the student is now mimicking R1) but wouldn't exceed it because the student's base is smaller and thus has less latent capability to surface. A more interesting direction: use R1 for the math data but distill from *other* teachers for non-math (different writing styles, different reasoning flavors), then do RL on the combined student. This is future work.

**21. Qwen-1.5B at 28.9 AIME — what's the limit?**

The paper doesn't test smaller. The implication is: reasoning can be *taught* to very small models via SFT on strong teacher outputs, provided the base model has sufficient representation capacity. Where does this break down? The limits are probably: (a) context length — 1.5B models may not fit 32K-token traces well; (b) vocabulary — small models have coarser tokenizers and may fragment numeric reasoning; (c) the "reasoning floor" where the base model can't represent the logical structure in its hidden states. Empirically, 1.5B works remarkably well. We'd guess the floor is somewhere below 500M but this is purely speculation.

**22. Teacher quality vs teacher diversity.**

A single strong teacher gives stylistic consistency, which helps the student avoid mode collapse. A mixed teacher corpus risks inconsistent reasoning styles. But mixed corpora might also mitigate the "distilled model inherits teacher's quirks" problem. For domains like math with clear correctness, single-teacher is probably optimal. For more open-ended tasks, mixed-teacher might reduce overfitting to one model's idiosyncrasies. No one has a great answer yet.

---

## Failed Approaches

**23. Why did PRMs fail?**

The three reasons are compelling in practice but not necessarily fundamental. "Define a step" is solvable with conventions (treat newlines or structural markers as step boundaries). Step-level annotation is noisy — but so is outcome annotation (for tasks without ground truth). Reward hacking of PRMs is the strongest argument: the policy exploits features of the PRM's scoring rather than actually improving its reasoning. Whether this is *solvable*: possibly with adversarial PRM training, frequent PRM retraining, or PRMs that are themselves trained adversarially against the policy. But all of these add complexity and cost, which is exactly what R1's philosophy rejects.

**24. MCTS failed because of branching factor.**

The response to this: Go's branching factor is ~250, but each step has a *well-defined, evaluable* board state. Language generation has a ~30K vocab branching factor *and* no canonical state — the "state" is the entire prefix. AlphaGo's value model was trained to predict game outcomes from board states; an LLM's value model would have to predict reasoning correctness from partial prefixes, which (as discussed in Q5) is much harder. The failure isn't fundamentally about branching — it's about the joint difficulty of (huge branching) × (weak value signal).

**25. What would it take to make tree search work for LLM reasoning?**

Plausible ingredients: (a) a *structured* action space (e.g., "emit a proof step" from a restricted grammar) rather than raw tokens; (b) a strong external verifier for each node (a theorem prover, a unit test); (c) a learned policy that narrows the branching factor at each node. Recent work on "tree of thoughts" and "MCTS for reasoning" (Yao et al.) explores this, though none have yet reached R1 performance. R1 essentially argues: if your external verifier is strong enough to power MCTS, it's also strong enough to provide RL signal directly, and RL is simpler.

---

## Safety, Limitations, and Practical Deployment

**26. How seriously should we take the safety claims?**

Safely deployed: R1 is roughly as safe as Claude-3.7 or o1, *with the risk control system*. Without it, R1 is notably worse on jailbreak benchmarks (85.9% unsafe). Key gaps: (1) the risk-control system is closed-source and not released with the weights, so open-weight users don't get it for free; (2) HarmBench failures (lyrics, IP) are a lawful-but-risky category the safety system doesn't address; (3) R1's multilingual safety wasn't tested against jailbreaks in non-English. A careful deployer would add their own content filter, rate limiting, and monitoring.

**27. Open weights + weak intrinsic safety = what obligation?**

This is a genuine dilemma and reasonable people disagree. The case for open release: R1's pretrained knowledge wasn't dangerous enough to withhold, and democratizing access has research value. The case for restriction: the model's reasoning capability could uplift novice users on reasoning-intensive harmful tasks. DeepSeek's compromise (open weights + documented risk-control recipe + ethics statement) is defensible but not uncontroversial. Alternatives: staged release (weights only to vetted researchers), license restrictions (non-commercial or non-weapons clauses — though hard to enforce), fine-tune-resistant safety (active research, no production system yet).

**28. Why does R1 underperform on SWE-Verified?**

The paper's explanation (long eval times limit RL steps) is partially convincing but probably incomplete. SWE-Bench requires *editing* a real codebase with multiple dependent files, which stresses different capabilities than isolated competitive programming: context management, understanding existing conventions, testing in a sandboxed environment. R1's reward signal for SWE tasks is "did the patch pass the hidden tests" — binary, sparse, and slow. Claude-3.5 was presumably SFT'd on large amounts of real-world code editing data. The fix is probably a combination of: more SE-focused RL data, async RL (overlap evaluation with next rollout), and SFT on high-quality real-world edits. The authors commit to this in future work.

**29. The few-shot degradation.**

This is genuinely surprising and the paper doesn't fully explain it. Three hypotheses: (a) R1's `<think>` format is strongly reinforced, and few-shot examples that don't match the template confuse the parser; (b) R1's reasoning strategy is well-calibrated for zero-shot prompts, and few-shot context adds noise that breaks its internal flow; (c) R1 has been trained to solve the task presented, not to complete a pattern established by examples — few-shot triggers the "complete the pattern" mode instead of the "reason about this problem" mode. Test: run few-shot with examples formatted in `<think>/<answer>` template and see if the degradation disappears.

**30. Adaptive test-time compute — is it really adaptive?**

Figure 18 shows a trend but not a clean signal — there's substantial variance, and even easy problems average ~7K tokens. Interpretation: R1 has a floor cost of ~5K tokens per problem (essentially what long CoT takes at minimum) plus difficulty-scaled additional compute up to ~18K. That's "somewhat adaptive" — it doesn't collapse to 100 tokens on trivial problems the way humans might. The authors acknowledge this in limitations ("instances of excessive reasoning—manifested as overthinking—are still observed in response to simpler questions"). For practical deployment this matters: R1 costs 10× more tokens than GPT-4o on *every* problem, including easy ones.

---

## Connections to Previous Weeks

**31. InstructGPT (W4) vs R1: has RLHF been superseded?**

Not superseded — generalized. InstructGPT proved RL from human feedback works for alignment. R1 proves RL from rule-based feedback works for reasoning. The two are complementary: R1's final stage uses human-feedback RMs for preference polish (inherited from InstructGPT's recipe) and rule-based rewards for reasoning (novel). The architecture is now "use the reward signal most appropriate to each capability dimension", which is a natural evolution of InstructGPT's framework rather than a replacement.

**32. Llama 2 (W5) used two RMs (helpfulness + safety); R1 does the same.**

The dual-RM pattern persists because the two objectives are in tension — maximizing helpfulness without safety produces harmful content, maximizing safety without helpfulness produces refusals. A single scalar reward forces the model to pick a point on the tradeoff curve at training time; two rewards let it trade off contextually. Llama 2's version learned a single policy with combined rewards; R1 uses the same formulation. Some recent work (Constitutional AI) tries to fold safety into a single RM with rule-based constraints, but the dual-RM pattern is battle-tested at scale.

**33. Llama 3 (W6) explicitly rejected MoE.**

Not incompatible — the two papers are solving different problems. Llama 3 cares about serving efficiency (dense is simpler to serve) and training stability at massive pretraining scale. R1 cares about RL-training efficiency where you need a large "effective capacity" base with modest activation cost per token (MoE is ideal). The two architecture choices optimize for different phases of the training lifecycle. A tempting synthesis: a dense pretrain followed by an MoE up-cycling for the RL phase. No one has published this yet, but it would arbitrage both arguments.

**34. GPT-3 (W1) argued for in-context learning at scale. R1 argues for in-weights reasoning at scale.**

These are genuinely in tension. GPT-3's few-shot prompting worked because the model's weights encoded many tasks but didn't specialize in any. R1's reasoning is baked into the weights because RL specialized the model hard. Another way to frame it: GPT-3 is a generalist that needs examples to specialize; R1 is a specialist that doesn't need examples at all. Whether R1's poor few-shot is fundamental or incidental: probably incidental — with enough RL on "follow in-context examples" prompts, you could likely restore few-shot ability. But it suggests specialization comes at a cost of context-adaptability.

**35. Mixture-of-Experts (W6).**

MoE's sparse activation means each RL gradient update only updates a subset of experts. Over many rollouts, different experts specialize to different reasoning patterns. This might actually *help* RL: the expert routing gives the model a built-in way to modularize reasoning strategies without explicit architecture changes. Conversely, expert load imbalance during RL could create optimization instabilities not seen in dense training. DeepSeek addresses this with their auxiliary-loss-free load balancing (from V3), but it's an open question whether this generalizes to other MoE architectures.

---

## Broader Questions

**36. The end of SFT-for-reasoning?**

For *training* reasoning models: probably yes, for large models with strong bases. For distilling reasoning into small models: SFT on R1 traces is now the recipe. So SFT shifts from "collect human reasoning demonstrations" to "sample from a strong RL-trained teacher." The humans who used to write CoT demonstrations are arguably out of the loop — unless their role moves to constructing *harder* reasoning problems that current models can't solve, or curating evaluation sets that aren't gameable.

**37. Reasoning as a separable axis of model capability.**

Yes — R1 demonstrates this fairly cleanly. Multilingual was the first modular capability (Llama 3.1's language expert branches); long-context was another (Llama 3's context-extension stage); reasoning is now one. The implication for model design: post-training becomes a pipeline of modular capability additions rather than a single alignment step. This aligns with the broader "foundation models as base + modular adaptations" paradigm.

**38. The cost-to-capability frontier.**

If R1's numbers generalize, then post-training is the new compute-efficient frontier. $294K for a reasoning model vs ~$6M for the base, and the reasoning model moves AIME from 39 to 80. But this only works if: (a) your base is already strong, (b) your reward is verifiable, (c) your infrastructure can handle long-rollout RL. The frontier isn't actually moving from scale to reward design — it's *adding* a new axis. The labs with both strong bases *and* good RL infrastructure (DeepSeek, OpenAI, Anthropic) continue to dominate; scale still matters for the base.

**39. Is the reasoning benchmark set fundamentally gameable?**

Yes and it's a real concern. AIME, MATH, Codeforces all have deterministic verifiers — precisely what R1 was trained against. R1's performance on benchmarks with subjective correctness (writing, strategy, design) is less well-characterized. AlpacaEval 2.0 (LC winrate 87.6) is one data point, but LLM-as-judge benchmarks have their own issues (length bias, style preference). The hard test would be: novel reasoning domains with expert human evaluation, not LLM evaluation, and not overlapping with training distribution. This is expensive and rare. Until such evaluations exist, "R1 reasons well" should be read as "R1 performs well on verifiable reasoning benchmarks."

**40. If pure RL works without SFT, what about without pretraining?**

Appendix G.1 already establishes that smaller bases don't RL-train into reasoning. So pure RL from a cold, untrained transformer seems very unlikely with current techniques. The base needs to be capable of *producing* plausible reasoning traces — without pretraining, it would only emit random tokens, and the RL signal would never find the narrow manifold of coherent language. An analogy: AlphaZero plays chess from scratch because the rules fully define the game and self-play provides dense structured feedback. Language has no such dense structure absent pretraining. The theoretical path forward would be environments where language generation has verifiable structure from the start (formal theorem proving, maybe), but general language RL-from-scratch seems a long way off.
