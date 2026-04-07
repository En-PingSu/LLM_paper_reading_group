# Week 6 — Discussion Questions
**Papers:** Mistral 7B (Jiang et al., Oct 2023) & Mixtral of Experts (Jiang et al., Jan 2024)

---

## Sliding Window Attention & Memory Efficiency

1. **Is a 4096-token window actually enough?** Sliding Window Attention limits each layer's attention to W=4096 tokens, but the paper claims an effective attention span of ~131K tokens through layered propagation across 32 layers. How realistic is this theoretical reach compared to true full attention? Can you construct a concrete task (e.g., long-range dependency resolution, multi-hop reasoning over a document) where the information at layer 0 would genuinely propagate through all 32 layers to reach tokens 131K positions away?

2. **Rolling buffer cache: an 8x memory reduction for free?** The rolling buffer cache overwrites position i with position i mod W, cutting KV cache size by 8x for the default context length. What kinds of generation tasks would suffer most from this fixed-window scheme? Consider a scenario where a user provides a 10K-token document and asks a question requiring information from the beginning — how does the model handle this compared to a full-attention model?

3. **Pre-fill chunking for long prompts.** Mistral 7B pre-fills the KV cache in chunks, where each chunk attends to itself and the cached window from prior chunks. What are the latency implications of this chunked approach versus processing the entire prompt at once? Does this strategy introduce any approximation errors relative to a single-pass attention computation?

4. **SWA as an inductive bias.** By forcing each layer to attend only locally, SWA implicitly assumes that long-range dependencies can be decomposed into chains of local interactions. Is this a reasonable assumption for natural language? Are there linguistic phenomena (e.g., garden-path sentences, long-distance anaphora, nested clauses) that genuinely require direct long-range attention and would be poorly served by layered propagation?

---

## Grouped-Query Attention

5. **GQA at 7B parameters — overkill or prescient?** LLaMA 2 only introduced Grouped-Query Attention at the 34B and 70B scales, yet Mistral 7B uses it at 7B (8 KV heads shared across 32 query heads, a 4:1 ratio). What motivated this choice at such a small scale? Is the throughput gain from reduced KV cache size more important at 7B than the potential quality loss from sharing key-value representations?

6. **The quality-throughput Pareto frontier.** GQA sits between Multi-Head Attention (every query head has its own KV head) and Multi-Query Attention (all query heads share one KV head). Mistral chose a 4:1 ratio. How would you design an experiment to find the optimal ratio for a given model size and deployment scenario? What metrics beyond perplexity matter when evaluating this tradeoff?

---

## Mixture of Experts Architecture

7. **46.7B params but only 12.9B active — is this fair?** Mixtral has 46.7B total parameters but only activates 12.9B per token. When the paper claims it "matches or outperforms LLaMA 2 70B," is this a fair comparison? Should we compare by total parameters, active parameters, FLOPs per token, memory footprint, or something else? What is the most honest way to characterize MoE model size?

8. **The gating function: how does Softmax(TopK(x * W_g)) learn specialization?** The router is a simple linear layer followed by softmax and top-k selection. There is no auxiliary loss explicitly encouraging expert specialization in the Mixtral paper. How does the router learn to distribute tokens meaningfully across experts? What prevents mode collapse (all tokens going to one or two experts)?

9. **Why 8 experts and top-2?** Mixtral uses 8 experts per layer and activates 2 per token. The paper does not ablate these choices. What would change with 16 experts and top-4 (same active compute), or 4 experts and top-1 (half the active compute)? How do you think the number of experts interacts with the granularity of specialization the model can learn?

10. **MoE memory burden.** Even though only 12.9B parameters are active per forward pass, you must load all 46.7B parameters into GPU memory. Under what deployment conditions does MoE actually save cost versus a dense 13B model? When does the memory overhead make MoE impractical, and what hardware configurations (e.g., multi-GPU, offloading) does it implicitly assume?

11. **Expert parallelism versus data parallelism.** MoE models are notoriously difficult to distribute across GPUs because different tokens route to different experts, creating irregular communication patterns. How does this affect the practical throughput advantage of Mixtral? Does the routing pattern (consecutive tokens often going to the same expert) help or hurt parallelism strategies?

---

## Routing Analysis & Expert Specialization

12. **Experts specialize syntactically, not by domain.** Section 5 of the Mixtral paper reveals that experts do NOT specialize by domain (ArXiv vs. GitHub vs. Wikipedia all use similar expert distributions). Instead, specialization appears to be syntactic — certain token types consistently route to certain experts. Why would syntactic specialization emerge over semantic or domain-level specialization? Is this a desirable property or a limitation?

13. **Consecutive token routing patterns.** The analysis shows that consecutive tokens in a sentence frequently route to the same expert. What does this suggest about what the router is learning — is it capturing something like phrase structure or n-gram patterns? How does this interact with the causal attention mask, given that the router makes independent decisions per token?

14. **"self" in Python always routes to the same expert.** Figure 8 shows that the Python keyword "self" is almost always assigned to the same expert. What does this tell us about the nature of expert specialization? Does the router essentially learn a soft tokenizer or POS-tagger? If experts learn syntactic roles, does this mean the model's semantic reasoning happens elsewhere (e.g., in the attention layers rather than the FFN experts)?

15. **Routing stability across layers.** Do the same tokens route to the same experts at every layer, or does the routing pattern change as information flows deeper through the network? If routing varies by layer, what does that imply about the hierarchical nature of what each layer's experts are learning?

---

## Efficiency & Scaling

16. **"Language models compress knowledge more than thought."** Mistral 7B matches LLaMA 2 13B on most benchmarks and approaches LLaMA 2 ~23B on MMLU, despite having far fewer parameters. What does it mean for a smaller model to "compress knowledge" more efficiently? Is there a fundamental limit to how much factual knowledge can be packed per parameter, or do architectural innovations like SWA and GQA unlock genuinely better compression?

17. **Active parameters as a scaling metric.** Mixtral matches LLaMA 2 70B with approximately 5x fewer active parameters. Should the field adopt active parameter count (or FLOPs per token) as the primary metric instead of total parameters? What are the failure modes of this metric — for example, does it ignore memory bandwidth, latency, or the cost of routing overhead?

18. **Three-dimensional scaling.** Both papers implicitly argue that scaling should be evaluated along three axes: capabilities, training cost, and inference cost. Traditional scaling laws (Kaplan et al., Hoffmann et al.) focus primarily on training compute. How should we extend scaling laws to account for inference efficiency, especially given that inference cost dominates for widely deployed models?

19. **Mistral 7B vs. LLaMA 2 13B: what accounts for the gap?** Mistral 7B outperforms LLaMA 2 13B despite having roughly half the parameters. The paper attributes this to architectural choices (SWA, GQA) but also likely benefits from better training data and longer training. How would you disentangle the contribution of architecture versus data versus training duration? What controlled experiments would you need?

---

## Instruction Tuning & Safety

20. **DPO over RLHF/PPO for alignment.** Both Mistral 7B Instruct and Mixtral Instruct use Direct Preference Optimization rather than PPO-based RLHF (as in InstructGPT from Week 4). What are the practical advantages of DPO for a startup like Mistral AI? Does DPO's simplicity come with any known limitations — for instance, in handling multi-turn dialogue or complex reward landscapes?

21. **System-prompt guardrails: 100% safe but less helpful.** With the recommended system prompt, Mistral 7B achieves 100% refusal rate on unsafe prompts but MT-Bench drops from 6.84 to 6.58 — a 4% hit to helpfulness. Is this tradeoff acceptable? More fundamentally, is system-prompt-based safety robust, or can it be trivially jailbroken compared to safety training baked into the model weights?

22. **Self-reflection for content moderation.** Mistral's "self-reflection" approach (the model judges whether its own output is safe) achieves 99.4% precision on the MT Bench harmful subset. But can a model reliably judge the safety of its own outputs? What are the adversarial attack vectors against a model-as-judge approach, and how does it compare to having a separate safety classifier?

23. **Mixtral Instruct vs. GPT-3.5-Turbo on Arena Elo.** At the time of release, Mixtral Instruct achieved a higher Arena Elo than GPT-3.5-Turbo and reportedly tied with Claude-2.1. How meaningful are these benchmarks for an open-weight model? Does the ability to fine-tune and customize an open model make static benchmark comparisons less relevant?

---

## Open-Source Strategy & Ecosystem

24. **Apache 2.0: truly open.** Both Mistral 7B and Mixtral are released under Apache 2.0, which imposes no restrictions on commercial use. How does this differ from Meta's LLaMA 2 community license (which restricts use above 700M monthly active users)? What are the strategic implications of choosing a fully permissive license for a VC-funded startup?

25. **Open-sourcing as a business model.** Mistral AI was founded in 2023 and quickly raised over 400M euros. How does giving away your core models for free generate revenue? Consider the Red Hat model, the MongoDB model, and the OpenAI pivot away from openness. Which trajectory is Mistral AI most likely following?

26. **Reproducibility and the minimal paper.** The Mistral 7B paper is 9 pages; Mixtral is 13. Neither provides detailed training data composition, hyperparameter schedules, or ablation studies. Is this level of detail sufficient for a scientific contribution, or are these closer to technical reports or product announcements? What is lost when model papers omit ablations?

---

## Connections to Previous Weeks

27. **GPT-3 (W1) to Mistral 7B: 25x smaller, arguably better.** GPT-3 has 175B parameters; Mistral 7B achieves competitive or better performance on many benchmarks with 25x fewer parameters. What combination of architectural advances (SWA, GQA, RoPE), training improvements (more tokens, better data), and evaluation differences account for this? Is it fair to say the field has achieved 25x compression, or are we measuring different capabilities?

28. **Attention Is All You Need (W2) meets Sliding Window Attention.** The original Transformer uses full quadratic attention over the entire sequence. SWA modifies this core mechanism to be linear in sequence length (for a fixed window). How does this relate to other efficient attention variants (Longformer, BigBird, Flash Attention)? Is SWA a simplification that works because of the layered propagation insight, or does it sacrifice something fundamental?

29. **InstructGPT's PPO (W4) vs. Mistral's DPO.** InstructGPT used a three-stage pipeline: SFT, reward model training, and PPO. DPO collapses the reward model and RL stages into a single supervised objective. Given what we discussed about reward hacking and instability in PPO (Week 4), does DPO avoid those problems or introduce new ones? Would InstructGPT have been better with DPO?

30. **LLaMA (W5) as the foundation.** Mixtral builds directly on the LLaMA architecture (RoPE, SwiGLU, RMSNorm, BPE tokenizer). The primary innovation is replacing dense FFN layers with sparse MoE layers. If LLaMA is the "backbone" and MoE is the "scaling trick," how much of Mixtral's performance should be attributed to each? Could you apply MoE to any dense Transformer and get similar gains?

---

## Broader Questions

31. **Is MoE the future of LLMs?** Since Mixtral, we have seen GPT-4 rumored to be MoE, Grok-1 confirmed as MoE, and DBRX using MoE. Is sparse MoE becoming the default architecture for frontier models? What are its fundamental limitations — routing instability, memory overhead, difficulty of fine-tuning — and could they prevent MoE from dominating?

32. **Dense vs. sparse: when does each win?** A dense 13B model and Mixtral (12.9B active from 46.7B total) have similar inference FLOPs per token, but very different memory and hardware requirements. Under what deployment scenarios (edge devices, cloud serving, fine-tuning, distillation) does each approach dominate? Is there a crossover point in model size where MoE becomes strictly better?

33. **The trend toward minimal publications.** Both papers are short, lack ablations, and were released on arXiv without peer review. Mistral 7B was initially released as just a magnet link on Twitter with no paper at all. Is this the future of ML research dissemination — models first, papers as afterthoughts? What does this mean for reproducibility and scientific rigor in the field?

34. **Efficiency versus capability frontiers.** Mistral and Mixtral optimize the Pareto frontier between cost and capability. But does optimizing for efficiency at a fixed capability level ultimately matter if frontier capabilities keep advancing? In other words, is it more impactful to make a 7B model match a 13B model, or to push the 13B model to match a 70B model? Who benefits most from efficiency gains?
