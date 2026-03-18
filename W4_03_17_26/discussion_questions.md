# Week 4 — Discussion Questions
**Paper:** Training Language Models to Follow Instructions with Human Feedback (InstructGPT), Ouyang et al. 2022

---

## The Alignment Problem

1. **Why doesn't scale solve alignment?** The paper shows a 1.3B InstructGPT is preferred over a 175B GPT-3. Why can't we just train a bigger model on more data to get helpful, honest, and harmless behavior? What is it about the next-token prediction objective that fundamentally limits this?

2. **Is "alignment" the right framing?** The paper frames the problem as aligning LMs to human intent. Could there be cases where users *want* the model to do something harmful? How should we think about alignment when user intent conflicts with broader societal values?

3. **How does InstructGPT compare to simply prompting GPT-3 well?** The paper tests GPT-3 with a carefully crafted few-shot prefix (GPT-3 prompted). Why isn't good prompting sufficient? What does RLHF add that prompting cannot?

---

## The RLHF Pipeline

### Step 1: SFT

4. **Why does the SFT model overfit on validation loss after 1 epoch but still improve with more training?** The authors train for 16 epochs despite overfitting after 1. They select models based on RM score, not validation loss. What does this tell us about the relationship between loss metrics and actual output quality?

5. **How important is the quality of the 13k demonstrations?** The SFT stage uses only ~13k examples — a tiny dataset compared to GPT-3's pretraining corpus. What would happen if the demonstrations were low quality or inconsistent? How sensitive is the pipeline to this first step?

6. **Why use human-written demonstrations rather than filtering/selecting from GPT-3's own outputs?** Could you skip SFT by simply selecting the best GPT-3 outputs to train a reward model directly?

### Step 2: Reward Model

7. **Why use a 6B reward model instead of 175B?** The paper mentions 175B RM training was unstable. What makes reward model training harder than language model training? What are the implications of having a reward model that is much smaller than the policy it evaluates?

8. **Why use pairwise rankings instead of absolute scores?** The RM is trained on "A is better than B" comparisons, not "A is a 7/10." What are the advantages of relative rankings over absolute ratings? What information is lost?

9. **What happens when labelers disagree?** Inter-annotator agreement is ~73%. That means labelers disagree about 27% of the time. How does the reward model handle this disagreement? Does it learn some kind of "average" preference, and is that meaningful?

10. **Walk through the RM loss function.** Why does the loss use sigmoid of the score *difference* rather than the raw scores themselves? What would go wrong if we just trained the RM to predict an absolute quality score?

### Step 3: PPO

11. **What is "reward hacking" and why is it a concern?** The KL penalty prevents the RL policy from drifting too far from the SFT model. Can you think of concrete examples of how a model might exploit the reward model to get high scores without actually being helpful?

12. **Why use a KL penalty rather than just capping the number of RL training steps?** What's the advantage of an explicit penalty term in the objective over simply stopping training early?

13. **Why is this a "bandit" environment rather than a full RL environment?** The model generates one complete response and gets one reward. How would things change if the model received per-sentence or per-token rewards instead?

---

## PPO-ptx and the Alignment Tax

14. **Is the alignment tax inevitable?** When we fine-tune a model to follow instructions, it gets worse at standard NLP benchmarks. Why does this happen? Is there a fundamental tradeoff between alignment and capability, or is it an artifact of the training procedure?

15. **Why does mixing pretraining gradients (PPO-ptx) work better than just increasing the KL coefficient?** The paper reports that increasing $\beta$ (the KL penalty) does not recover benchmark performance as well as PPO-ptx. Why might these two approaches have different effects?

16. **Could the alignment tax be measured differently?** The paper measures it on benchmarks like HellaSwag and SQuAD. But if InstructGPT is better at following instructions, might it actually perform *better* on these benchmarks if prompted in the right way?

---

## Evaluation

17. **Is "helpful, honest, harmless" the right framework?** These three criteria can conflict — a maximally helpful model might comply with harmful requests. How should we prioritize among them? The paper prioritizes helpfulness during training but truthfulness/harmlessness during evaluation. Is this the right balance?

18. **Why didn't RLHF reduce bias?** InstructGPT improves on truthfulness and toxicity but shows no improvement on Winogender and CrowS-Pairs (bias benchmarks). Why might RLHF help with some alignment dimensions but not others? What would it take to reduce bias through this pipeline?

19. **How reliable is the TruthfulQA benchmark?** InstructGPT is ~2x more truthful than GPT-3 on TruthfulQA. But TruthfulQA contains adversarial questions designed to trick models. Does performance on adversarial questions tell us about truthfulness in normal usage?

20. **What does a 21% vs 41% hallucination rate actually mean?** On closed-domain tasks, InstructGPT hallucinates about half as often as GPT-3. Is 21% still too high for real-world deployment? What kinds of hallucinations remain, and how might they be further reduced?

---

## Who Are We Aligning To?

21. **Who decides what "aligned" means?** The labelers are ~40 English-speaking contractors. How would InstructGPT behave differently if labelers came from a completely different cultural background? Is "alignment to labeler preferences" a meaningful goal?

22. **How do OpenAI's labeling instructions influence the outcome?** The labelers follow researcher-written guidelines. To what extent is InstructGPT aligned to the labelers vs. aligned to OpenAI's vision of what good behavior looks like?

23. **Is it possible to align a single model to everyone?** The paper acknowledges that different groups have different preferences. Is a single aligned model feasible, or do we need models that can be conditioned on the preferences of specific communities?

24. **What are the implications of training on API customer prompts?** The training data comes primarily from OpenAI API users. These users were selected from a waitlist seeded by OpenAI's networks. How does this data source bias what the model learns to value?

---

## Connections to Previous Weeks

25. **How does InstructGPT relate to GPT-3's few-shot learning (W1)?** GPT-3 showed that larger models are better few-shot learners. InstructGPT shows that a smaller aligned model can beat a larger unaligned one. Are these findings in tension, or complementary?

26. **How does the SFT stage relate to GPT-1's fine-tuning approach (W3)?** GPT-1 introduced the pretrain-then-fine-tune paradigm. How is InstructGPT's SFT step similar to and different from GPT-1's fine-tuning?

27. **Could RLHF be applied to GPT-2 or GPT-1?** The paper uses GPT-3 as the base model. Would the same RLHF pipeline work on smaller models from earlier weeks? What might change?

---

## Broader Questions

28. **What has changed since this paper was published (2022)?** Modern systems like ChatGPT, Claude, and Gemini all build on RLHF or similar techniques. What improvements have been made? What problems identified in this paper remain unsolved?

29. **Is RLHF the best approach to alignment?** Are there alternatives to RLHF (e.g., constitutional AI, direct preference optimization / DPO, debate) that might address some of InstructGPT's limitations? What are the tradeoffs?

30. **What are the risks of making models better at following instructions?** The paper briefly notes that InstructGPT generates *more* toxic outputs than GPT-3 when explicitly prompted to be toxic. Is a more instruction-following model inherently more dangerous if given harmful instructions?
