# Week 8 — Discussion Questions
**Paper:** DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning, DeepSeek-AI, January 2025

---

## The "Pure RL" Thesis

1. **Does R1-Zero actually prove reasoning is incentivized rather than taught?** The paper's strongest claim is that long CoT *emerges* from RL rather than being *copied* during SFT. But DeepSeek-V3-Base was pretrained on web text that includes reasoning-heavy documents (math textbooks, Stack Overflow, olympiad solutions). How can we distinguish "RL incentivizes latent behavior" from "RL surfaces a capability that was already in the weights"? What experiment would falsify the stronger claim?

2. **The aha moment — genuine emergence or selection bias?** Table 2 shows one cherry-picked trace where the model writes "That's an aha moment I can flag here." The paper doesn't report how frequent these moments are, or whether they ever help versus hurt accuracy. If 1% of traces have "aha moments" and they correlate with correctness, is that convincing? What would you want the authors to measure?

3. **Why does response length grow?** Figure 1 shows response length monotonically rising from ~2K to ~15K tokens. Nothing in the reward function directly rewards length. Walk through the mechanism: why would RL push the model to produce longer outputs? Could this be a failure mode dressed up as a feature — e.g., the model just learns that *any* extra tokens reduce variance on its predictions?

4. **What does "no SFT" really mean?** R1-Zero skips SFT on reasoning traces, but DeepSeek-V3-Base itself had a post-training cooldown phase and was pretrained on 14.8T tokens including reasoning content. Is R1-Zero "pure RL from raw pretraining" or "RL applied to a model that already implicitly knows how to reason"? Does the distinction matter for the paper's thesis?

---

## GRPO and the RL Algorithm

5. **Why does dropping the value model work?** PPO's value model estimates expected return from a partial trajectory. GRPO estimates the advantage *purely* from G=16 sampled full trajectories on the same prompt. What's the statistical tradeoff? When would you expect GRPO to fail relative to PPO?

6. **Is GRPO actually novel?** GRPO is from DeepSeekMath (Shao et al., 2024), not original to this paper. The paper applies it to reasoning specifically. Does the choice of RL algorithm materially change the scientific claim, or would PPO with a (poorly-trained) value model reach similar conclusions?

7. **Group size as a hyperparameter.** GRPO samples G=16 outputs per prompt. What's the effect of G? At G=2, advantages would be {+1, -1} almost always — essentially binary preference. At G=64, the advantage becomes a fine-grained ordering. Does G trade off exploration (more samples → more diverse outcomes to compare) against compute (more samples → slower training)?

8. **The clip ratio of 10.** Stage 1 RL uses $\varepsilon{=}10$, not the usual 0.2. The paper says smaller clip values "truncate gradients for a significant number of tokens." What does this imply about how GRPO behaves on long outputs, and is this a sign the objective isn't quite well-conditioned?

9. **Why is KL divergence so low ($\beta{=}0.001$)?** InstructGPT used $\beta$ in the 0.01–0.1 range. R1's 100× weaker KL essentially lets the policy drift freely from the reference. Is that safe when the only reward is accuracy? Could you construct a scenario where this gets you into trouble?

---

## Reward Design

10. **Why reject neural reward models for reasoning?** The paper says "neural reward models are susceptible to reward hacking during large-scale RL" and uses only rule-based rewards. But this limits the approach to domains with verifiable ground truth. How broadly does this really generalize — is R1's recipe fundamentally limited to math, code, and logic?

11. **Accuracy + format, equally weighted.** The rule-based reward is $\mathrm{Reward}_{\mathrm{acc}} + \mathrm{Reward}_{\mathrm{format}}$ with equal weights. Why not weight accuracy higher? What happens if format reward alone gets the model's attention first (optimizing for structure before correctness)?

12. **The language consistency reward costs accuracy.** Figure 7 shows applying the LC reward reduces math benchmark performance by ~1–2 points. The authors accept this as "aligning with human preferences." Is this a principled trade-off or a signal that something is wrong — e.g., is the model's best reasoning genuinely easier in a language-mixed form?

13. **Format reward as structural prior.** The `<think>...</think><answer>...</answer>` template is both a reward target and a parsing convenience. Does it also implicitly *scaffold* the model's behavior — i.e., would R1-Zero's emergent reasoning happen without the format constraint?

---

## The Multi-Stage Pipeline

14. **Why does cold-start SFT *hurt* AIME?** Table 3 shows R1-Dev1 (after cold-start SFT) drops AIME from 77.9 (R1-Zero) to 59.0. The small SFT dataset seems to overwrite R1-Zero's reasoning. If cold start hurts reasoning, why include it? What makes the eventual Dev2/Dev3/R1 recovery possible?

15. **Why not "R1-Zero then cold-start then RL" in a single pipeline?** The paper starts cold-start from V3-Base, not from R1-Zero. Why throw away R1-Zero's weights and restart? Is this a principled choice or an artifact of experimental convenience?

16. **Four stages — is this the minimum?** The R1 pipeline is Cold-Start SFT → Reasoning RL → Rejection SFT → General RL. Could you ablate any one stage? For a research replication, which stages are load-bearing and which are polish?

17. **Using DeepSeek-V3 as judge.** Stage 2 uses DeepSeek-V3 as an LLM-judge to filter rejection-sampled reasoning data. But V3 is strictly *less capable* at reasoning than R1-Dev2 (which generates the data). How does a weaker judge reliably filter a stronger model's outputs?

18. **General RL only moves the preference benchmarks.** Table 3 shows Stage 3 moves AlpacaEval by +25 and ArenaHard by +17, but reasoning benchmarks move by <1 point. Is General RL orthogonal to reasoning, or is it subtly degrading reasoning that rejection SFT then masks?

---

## Distillation vs RL

19. **Distill-32B beats Qwen-32B-Zero by 25 AIME points.** Table 16 shows a 32B model trained with the same RL recipe as R1-Zero scores 47.0 on AIME, while the same base distilled from R1 scores 72.6. Why does distillation beat RL so dramatically at this scale? Is it that small models can't find good reasoning strategies, or that big-teacher traces encode strategies the small model couldn't have found?

20. **When should you run RL on top of a distilled model?** The paper applies only SFT to the distilled models and flags RL as future work. Would RL on the distilled models just recover R1's ceiling, or could a distilled + RL'd model *surpass* R1 by combining both signal sources?

21. **Qwen-1.5B at 28.9 AIME — what's the limit?** A 1.5B model scoring 3× GPT-4o on AIME is striking. What does this imply about the *teachability* of reasoning via SFT? Is there a "reasoning floor" model size below which even distillation fails?

22. **Teacher quality vs teacher diversity.** R1's 800K distillation set comes from one teacher. Would a mixed teacher corpus (R1 + o1 + Claude) produce better students, or would it inject conflicting reasoning styles?

---

## Failed Approaches

23. **Why did PRMs fail?** The paper lists three reasons PRMs didn't work: hard to define "a step", noisy step-level supervision, and reward hacking. Are any of these problems solvable, or is step-level reward inherently a dead end for reasoning?

24. **MCTS failed because of branching factor — but so did AlphaGo.** AlphaGo's branching factor (~250 in Go) is already huge and MCTS handled it. Why does the 30K-vocab branching factor of NLP make MCTS infeasible, when Go's search space is astronomically larger overall?

25. **What would it take to make tree search work for LLM reasoning?** The paper gives up on MCTS too quickly in our view — is there a hybrid (smaller action space, learned search heuristics) that could revive it?

---

## Safety, Limitations, and Practical Deployment

26. **How seriously should we take the safety claims?** R1 scores 95.0 on HELM safety (Table 9) but HarmBench shows 35% unsafe due to IP/lyrics issues. With the risk control system, unsafe rate drops to 8.5%. If you were deploying R1, what additional safety infrastructure would you want beyond what's described?

27. **Open weights + weak intrinsic safety = what obligation?** The ethics statement acknowledges R1 can be fine-tuned to remove safety. Is it responsible to release open weights under MIT when the base model's jailbreak rate without risk control is 85.9%? What are the alternatives — fine-tune-resistant safety, staged release, license restrictions?

28. **Why does R1 underperform on SWE-Verified?** R1 scores 49.2 on SWE-Verified vs Claude-3.5-Sonnet's 50.8, despite winning on almost every other benchmark. The paper blames "long evaluation times" for RL. Is this a fixable engineering problem or a deeper issue with RL for code tasks?

29. **The few-shot degradation.** R1 performs *worse* with few-shot prompting. This is the opposite of GPT-3's central finding. Why? Is the issue specifically that few-shot examples conflict with R1's trained `<think>` format, or is there something deeper about how reasoning-RL'd models process context?

30. **Adaptive test-time compute — is it really adaptive?** Figure 18 shows token use scales with problem difficulty, but the signal is noisy. On simple problems R1 still uses ~7K tokens. Is R1 genuinely adaptive, or does it have a roughly flat minimum cost per problem plus variance that happens to correlate with difficulty?

---

## Connections to Previous Weeks

31. **InstructGPT (W4) vs R1: has RLHF been superseded?** InstructGPT's pipeline was SFT → RM → PPO. R1's final Stage 3 is SFT → (rule+RM) → GRPO. The architecture is nearly identical, but R1 treats RMs as a helpfulness-polish step while *reasoning* is done by rules alone. Has InstructGPT's recipe been generalized or partially deprecated?

32. **Llama 2 (W5) used two RMs (helpfulness + safety); R1 does the same in Stage 3.** This dual-RM pattern has persisted across two very different pipelines. Why is it so robust? Is it a principled design or a historical accident?

33. **Llama 3 (W6) explicitly rejected MoE in favor of dense for training stability.** R1 explicitly *requires* MoE (DeepSeek-V3) — their ablation found that small dense models couldn't learn reasoning via RL. Are these two positions compatible, or does one camp have to be wrong? Does RL training have different architectural requirements than pretraining?

34. **GPT-3 (W1) argued for in-context learning at scale. R1 argues for in-weights reasoning at scale.** Are these rival paradigms, or is R1's poor few-shot behavior just a side effect of a training choice that could be undone?

35. **Mixture-of-Experts (W6).** DeepSeek-V3-Base is a 671B/37B MoE. RL rollouts on MoE have a peculiar property: only a subset of experts fire per token, so the gradient updates are sparse. Does this change the RL dynamics in any way that matters for R1's success?

---

## Broader Questions

36. **The end of SFT-for-reasoning?** If R1-Zero works and distillation works, is there any role left for SFT on human-written reasoning traces? Who should still be writing CoT demonstrations?

37. **Reasoning as a separable axis of model capability.** R1 demonstrates that you can add reasoning to a base model via post-training without changing architecture. Should we now think of "reasoning" as a modular capability (like multilingual, long-context) rather than an emergent property of scale?

38. **The cost-to-capability frontier.** R1's disclosed training cost is $294K. DeepSeek-V3-Base presumably cost ~$6M. Scaling pretraining 10× from here is ~$60M; scaling RL 10× is ~$3M. If RL gains keep compounding, does the frontier shift from data/params to reward design?

39. **Is the "reasoning" benchmark set fundamentally gameable?** AIME, MATH, LiveCodeBench, Codeforces all have deterministic verifiers — exactly what R1 is trained to optimize. Does R1's benchmark success generalize to reasoning in *unverifiable* domains (legal analysis, strategic planning, design)? How would we know?

40. **If pure RL works without SFT, what about without pretraining?** The paper concedes small models can't do RL from scratch, but DeepSeek-V3-Base is already heavily pretrained. Is there a regime — bigger models, better RL — where we could skip supervised pretraining and train directly via RL on verifiable rewards? What would that even look like?
