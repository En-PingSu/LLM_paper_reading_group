# Week 5 — Discussion Questions
**Papers:** LLaMA: Open and Efficient Foundation Language Models (Touvron et al., Feb 2023) & Llama 2: Open Foundation and Fine-Tuned Chat Models (Touvron et al., Jul 2023)

---

## Scaling & Efficiency (LLaMA 1)

1. **Training budget vs. inference budget — who wins?** LLaMA argues we should optimize for inference cost, not training cost. Chinchilla says a 10B model on 200B tokens is compute-optimal for training, but LLaMA trains a 7B model on 1T tokens (5x "over-trained"). When is it worth paying more to train if it saves on inference? Are there scenarios where optimizing for training budget is still the right call?

2. **Can you train forever?** LLaMA-7B's training loss is still decreasing at 1T tokens (Figure 1). Does this mean we should just keep training? Is there a theoretical limit to how much a 7B model can improve with more data? What would we expect to happen at 10T or 100T tokens?

3. **Does LLaMA make Chinchilla wrong?** Chinchilla's scaling laws predict the "optimal" model size and data for a given compute budget. LLaMA deliberately violates these by training smaller models on much more data. Are the Chinchilla scaling laws wrong, or is LLaMA just answering a different question?

---

## Open-Source vs. Closed-Source

4. **What does "only public data" actually mean?** LLaMA claims to use only publicly available data, but CommonCrawl (67% of training data) contains copyrighted material, personal information, and content people didn't consent to be used for AI training. Is "publicly available" the same as "ethically unproblematic"? How does this compare to GPT-3's use of proprietary datasets?

5. **Is open-sourcing model weights responsible?** LLaMA 1 was research-only; Llama 2 allows commercial use. What are the risks and benefits of releasing model weights? The Llama 2 paper argues open release promotes safety through community scrutiny. Critics argue it enables misuse. Who's right?

6. **Did LLaMA change the industry?** LLaMA's leaked weights catalyzed models like Alpaca, Vicuna, and an explosion of open-source fine-tuning. Did this accelerate AI capabilities dangerously, or did it democratize access beneficially? What would the landscape look like if LLaMA hadn't been released?

---

## Architecture Choices

7. **Why do RMSNorm, SwiGLU, and RoPE matter?** These three modifications to the original Transformer are now standard in virtually all modern LLMs. For each: what specific problem does it solve, and why did it take 5+ years after the original Transformer paper (2017) to adopt them?

8. **Why use Grouped-Query Attention only for 34B+?** Llama 2 uses GQA for the 34B and 70B models but not 7B and 13B. What determines whether GQA is worth using? What is the quality vs. speed tradeoff, and how would you decide the cutoff for a new model?

9. **Is the architecture still the bottleneck?** Both papers use essentially the same decoder-only transformer. If the architecture is "solved," what remains the bottleneck for performance — data, compute, or alignment?

---

## Training Data & Data Curation

10. **CommonCrawl is 67% of LLaMA's diet. Is this a problem?** Two-thirds of the training data comes from web crawl data, despite extensive filtering. What kinds of biases and failure modes might this introduce? How does the quality filtering (fastText classifier trained on Wikipedia references) shape what the model learns?

11. **Why does Wikipedia get 2.45 epochs but CommonCrawl only 1.10?** The data mixing strategy upsamples high-quality sources. What are the tradeoffs of seeing the same data multiple times? Could this cause memorization? How would you decide the optimal sampling ratio?

12. **Llama 2's pretraining data is undisclosed.** Unlike LLaMA 1 (which described its data sources in detail), Llama 2 only says "a new mix of publicly available online data." Why might Meta have been less transparent about data composition? Does this undermine the "open" claim?

---

## Fine-tuning Pipeline (Llama 2)

13. **"Quality is all you need" — is 27k SFT examples really enough?** Llama 2 found that tens of thousands of high-quality annotations beat millions of low-quality ones. Why might this be? Is there a minimum threshold, and what happens below it? How does this compare to InstructGPT's 13k demonstrations?

14. **Why does Llama 2 compute loss only on answer tokens?** During SFT, the loss is masked for prompt tokens. InstructGPT appears to compute loss on the full demonstration. What's the advantage of answer-only loss? Could including prompt tokens in the loss help or hurt?

15. **Why two reward models instead of one?** InstructGPT used a single RM. Llama 2 separates helpfulness and safety into two models. What specific failure modes does this address? Could you use three or more reward models (e.g., separating truthfulness)? What are the limitations of this approach?

16. **What does the margin $m(r)$ in the RM loss actually buy you?** Llama 2's RM loss includes a margin based on preference strength ("significantly better" vs "slightly better"). How does this change what the RM learns compared to InstructGPT's standard loss? Is the improvement worth the annotation complexity?

17. **Iterative RLHF: why 5 rounds?** Llama 2 applies RLHF in 5 iterations (V1–V5). Why not 1 round like InstructGPT? Why not 10 or 20 rounds? Is there a point of diminishing returns? What determines when to stop iterating?

18. **Rejection Sampling vs. PPO — why use both?** Rejection Sampling provides "breadth" (exploring K samples per prompt); PPO provides "depth" (each step builds on the previous). Why combine them? Could you achieve the same result with just one approach but more iterations?

---

## Ghost Attention (GAtt)

19. **Why do RLHF models forget the system message?** Before GAtt, Llama 2-Chat would stop following system instructions after a few dialogue turns. Why does this happen? Is it an attention mechanism limitation, or a consequence of how the training data is structured?

20. **Is GAtt a hack or a principled solution?** GAtt works by synthetically concatenating the system message to all user turns during data collection, then masking the loss on intermediate assistant messages. Is this a fundamental fix, or a workaround that could fail in edge cases?

---

## Safety & Alignment

21. **Is 0% toxicity on ToxiGen actually good?** Llama 2-Chat achieves effectively 0% toxicity on ToxiGen. But does this mean the model is truly safe, or has it learned to avoid generating text that a classifier flags as toxic? Could it still produce harmful content in ways that bypass toxicity classifiers?

22. **Context distillation: teaching the model to be safe without being told.** The model generates safer outputs when given a safety preprompt, then is trained to produce those outputs without the preprompt. Does this really change the model's "values," or does it just hide the capability behind a behavioral mask?

23. **The false refusal problem.** Llama 2-Chat sometimes refuses legitimate prompts that contain sensitive-looking words (e.g., "Christmas Crack" recipe). Is this an inevitable consequence of safety tuning? How would you measure and minimize false refusals while maintaining safety?

24. **350 red teamers — is that enough?** The paper describes red teaming by 350+ people across diverse backgrounds. How do you know when you've done enough red teaming? What kinds of risks might a large red team still miss? How does red teaming scale with model capability?

25. **Safety data scaling: can you have too much safety data?** Figure 15 shows that safety improves with more safety data while helpfulness stays constant. Is this always true? At what point might safety data start hurting helpfulness (over-refusal)? The paper measures a 0.05% false refusal rate — is this an underestimate?

---

## Evaluation & Benchmarks

26. **Human evaluation vs. model-based evaluation.** Llama 2 uses both human raters and GPT-4 as a judge. When should we trust human evaluation over model-based evaluation? Can GPT-4 evaluate a model that might eventually surpass it?

27. **Is "competitive with ChatGPT" meaningful?** Llama 2-Chat 70B has a 36% win rate vs ChatGPT (with 31.5% ties). The paper frames this as "competitive." Is a 36% win rate competitive, or is ChatGPT clearly better? How should we interpret win-rate comparisons?

28. **MMLU as the universal benchmark.** Both papers heavily rely on MMLU. LLaMA-65B scores 63.4; Llama 2-70B scores 68.9; GPT-4 scores 86.4. Is MMLU a good measure of model quality? What does the 18-point gap between Llama 2-70B and GPT-4 actually represent in practical terms?

---

## Connections to Previous Weeks

29. **From GPT-3 to LLaMA: what changed and what stayed the same?** GPT-3 (W1) and LLaMA share the same fundamental architecture (decoder-only transformer). LLaMA adds RMSNorm, SwiGLU, and RoPE. Is LLaMA a qualitatively different model, or just GPT-3 with better engineering?

30. **From InstructGPT to Llama 2-Chat: is this progress or iteration?** Llama 2-Chat follows InstructGPT's SFT → RM → PPO pipeline with additions (two RMs, rejection sampling, iterative rounds, GAtt). Are these fundamental innovations or incremental improvements? What would a truly different approach look like?

31. **The full arc: Transformer → GPT-1/2 → GPT-3 → InstructGPT → LLaMA/Llama 2.** Looking at the full sequence we've read, what has been the single most important advance? Is it the architecture (W2), the scale (W1), the alignment (W4), or the open release (W5)?

---

## Broader Questions

32. **Is the open-source model gap closing?** At the time of writing (Jul 2023), Llama 2-70B was competitive with ChatGPT but far behind GPT-4. As of early 2026, where does this gap stand? What factors determine whether open-source models can catch up?

33. **What did Llama 2 get wrong?** Every paper looks impressive when read fresh. In hindsight (with Llama 3 and other subsequent work), what decisions in the Llama 2 paper turned out to be suboptimal? What approaches were abandoned in later work?

34. **RLHF vs. DPO vs. other alignment methods.** Llama 2 uses the classic RLHF pipeline (reward model → PPO). Since this paper, Direct Preference Optimization (DPO) and other alternatives have gained popularity. What are the tradeoffs? Why might Meta have stuck with RLHF?

35. **The emergence question.** Llama 2-Chat shows emergent temporal perception and tool use without explicit training. Are these genuine emergent capabilities, or artifacts of the training data and alignment process? How should we think about emergence in aligned models?
