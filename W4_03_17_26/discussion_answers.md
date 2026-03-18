# Week 4 — Discussion Questions & Suggested Answers
**Paper:** Training Language Models to Follow Instructions with Human Feedback (InstructGPT), Ouyang et al. 2022

These are suggested answers to guide discussion, not definitive answers. Many of these questions are deliberately open-ended.

---

## The Alignment Problem

**1. Why doesn't scale solve alignment?**

The next-token prediction objective trains the model to mimic the statistical distribution of its training data — which includes misinformation, toxic content, unhelpful text, and every writing style on the internet. A bigger model gets *better* at mimicking that distribution, but "better at predicting internet text" is not the same as "more helpful to the user." A 175B model might produce a more fluent conspiracy theory than a 1.3B model. Scale improves *capability* (what the model *can* do) but not *alignment* (what the model *should* do). RLHF addresses the gap by introducing a separate signal — human preferences — that directly encodes what users actually want.

**2. Is "alignment" the right framing?**

This is genuinely debated in the field. The paper acknowledges the tension: during training, they prioritize helpfulness (following user instructions), but a user could ask for something harmful. The paper's practical stance is to align to labeler preferences shaped by researcher guidelines — not to some abstract notion of "human values." Some researchers argue for a hierarchy: harmlessness > honesty > helpfulness. Others argue we need models that can *refuse* instructions, meaning alignment isn't just "do what the user wants" but "do what's appropriate." There's no consensus — this is an active area of research.

**3. How does InstructGPT compare to simply prompting GPT-3 well?**

The paper tests this directly — "GPT-3 (prompted)" uses a carefully crafted few-shot prefix. Results show it performs better than vanilla GPT-3 but significantly worse than SFT and PPO models. Prompting has fundamental limitations:
- It uses up context window space that could be used for the actual task
- It's fragile — small prompt changes can dramatically alter behavior
- It can't change the model's underlying probability distribution, only steer it
- RLHF actually updates the model weights, creating durable behavioral changes that generalize to new prompts the model hasn't seen

---

## The RLHF Pipeline

### Step 1: SFT

**4. Why does the SFT model overfit on validation loss after 1 epoch but still improve with more training?**

This reveals an important disconnect: validation loss (cross-entropy on held-out data) measures how well the model predicts the exact tokens in the demonstrations. But "matching the exact tokens" and "producing responses humans prefer" are different goals. After 1 epoch, the model has memorized the demonstrations well enough that validation loss stops improving. But continued training makes the model's *style* and *behavioral patterns* more consistent — it becomes more reliably helpful, follows instructions more consistently, and produces outputs the RM scores higher. This is why the authors use RM score (a proxy for human preference) rather than validation loss for model selection. It's a concrete example of Goodhart's Law: optimizing for the metric (validation loss) doesn't necessarily optimize for the goal (human preference).

**5. How important is the quality of the 13k demonstrations?**

Very important — the demonstrations define the "target behavior" for the entire pipeline. They establish the model's style, tone, and approach to following instructions. Low-quality demonstrations would propagate errors through the entire pipeline: the SFT model would learn bad habits, the RM would be trained to compare outputs against a flawed baseline, and PPO would optimize toward a flawed reward signal. That said, 13k is remarkably small — it works because the model already has strong language capabilities from pretraining. The demonstrations don't teach the model language; they teach it *how to behave*.

**6. Why use human-written demonstrations rather than filtering GPT-3's own outputs?**

Human-written demonstrations can show behaviors that GPT-3 rarely produces on its own — like politely declining harmful requests, asking clarifying questions, or admitting uncertainty. If you only filtered GPT-3 outputs, you'd be limited to the behaviors GPT-3 already exhibits, which are biased toward the distribution of internet text. Human demonstrations can introduce genuinely new behavioral patterns. That said, later work (e.g., Constitutional AI) has explored using model-generated data more heavily, suggesting this is a spectrum rather than a binary choice.

### Step 2: Reward Model

**7. Why use a 6B reward model instead of 175B?**

The paper reports that 175B RM training was unstable. Reward model training is harder than language model training because:
- The training signal is noisier — human preferences are subjective and inconsistent (~73% agreement)
- The loss landscape may be more complex since the model must learn to compare outputs rather than predict tokens
- Overfitting is a bigger risk with relatively small comparison datasets (33k rankings)

The implication of a smaller RM evaluating a larger policy is interesting: the RM may not capture all the nuances that the 175B policy can express. This creates a potential ceiling — the policy can only be as good as the RM can evaluate. In practice, the 6B RM works well enough because preference judgments may be simpler than language generation.

**8. Why use pairwise rankings instead of absolute scores?**

Relative comparisons are easier and more reliable for humans than absolute ratings. If you ask "rate this response 1-10," different people calibrate their scales differently (one person's 7 is another's 5). But if you ask "is A better than B?", people are much more consistent. This is well-established in psychometrics.

What's lost: intensity information. If A is only slightly better than B, a pairwise ranking treats it the same as A being far better than B. The RM loss function partially addresses this — the sigmoid naturally produces smaller gradients for close comparisons — but the raw ranking doesn't encode magnitude.

**9. What happens when labelers disagree?**

The RM effectively learns a weighted average of labeler preferences. When labelers consistently disagree on a type of prompt (e.g., politically sensitive topics), the RM learns to assign middling scores — neither strongly positive nor negative. This "averaging" effect can wash out strong minority preferences.

The ~73% agreement rate means roughly 1 in 4 comparisons are contested. The paper mitigates this somewhat by using multiple comparisons per prompt (K=4 to K=9 responses), which provides redundant signal. But fundamentally, a single scalar reward cannot capture the diversity of human preferences — this is one of the paper's acknowledged limitations.

**10. Walk through the RM loss function.**

The loss uses sigmoid of the score *difference* rather than raw scores because:
- **Translation invariance:** Only the *difference* between scores matters, not their absolute values. Adding a constant to all scores doesn't change the loss. This is desirable because we care about rankings, not absolute quality levels.
- **Calibrated probabilities:** The sigmoid converts the score difference into a probability in (0,1) that the preferred response is ranked higher. This connects directly to the Bradley-Terry model of pairwise preferences from psychometrics.
- **Smooth gradients:** If we used raw scores with a margin-based loss (e.g., "r_w should be at least 1 point higher than r_l"), the gradient would be zero once the margin is met, stopping learning. The sigmoid always provides a gradient, pushing the model to be increasingly confident.

If we trained the RM to predict absolute quality scores, we'd need labeled scores (not just rankings), and different labelers would calibrate differently, making the data much noisier.

### Step 3: PPO

**11. What is "reward hacking" and why is it a concern?**

Reward hacking occurs when the RL policy finds ways to get high reward scores without actually being helpful. The RM is an imperfect proxy for human preferences — it has blind spots. Without constraints, the policy will exploit those blind spots. Concrete examples:
- Generating very long, verbose responses (if the RM slightly favors longer outputs)
- Repeating confident-sounding phrases or filler that the RM associates with quality
- Using flattering or sycophantic language
- Producing outputs that pattern-match to high-scoring training examples without meaningful content

The KL penalty prevents this by keeping the policy close to the SFT model — if the policy starts generating unusual text to game the RM, the KL penalty increases, reducing the objective.

**12. Why use a KL penalty rather than just capping the number of RL training steps?**

Early stopping is a blunt instrument — it limits *all* learning, both beneficial and harmful. The KL penalty is more surgical: it allows the model to learn behaviors that the SFT model already somewhat supports (small KL) while strongly penalizing behaviors that deviate far from the SFT distribution (large KL). This means:
- Common, high-quality responses can be reinforced freely
- Unusual, potentially reward-hacked responses are penalized proportionally to how unusual they are
- The penalty is per-token, giving fine-grained control

Early stopping would require careful tuning and would produce a different tradeoff for every prompt type.

**13. Why is this a "bandit" environment rather than a full RL environment?**

In a full RL environment, the agent takes a sequence of actions across multiple states, receiving intermediate rewards. Here, the model generates one complete response and gets one reward — there are no intermediate states or rewards. This is simpler because:
- The RM is trained to evaluate complete responses, not partial ones
- It avoids the credit assignment problem (which token caused the high/low reward?)
- It's computationally cheaper (one RM forward pass per response)

Per-token rewards would be more informative but much harder to obtain — you'd need humans to rate partial responses, or a much more sophisticated RM. Some later work explores token-level rewards, but the bandit approach was practical and effective for this paper.

---

## PPO-ptx and the Alignment Tax

**14. Is the alignment tax inevitable?**

It likely reflects catastrophic forgetting — when fine-tuning on a new objective (follow instructions), the model partially "forgets" capabilities learned during pretraining. This is a well-known phenomenon in ML. However, PPO-ptx shows it's largely mitigable by mixing pretraining data into the RLHF training. This suggests the alignment tax is more of an engineering challenge than a fundamental tradeoff. The model has the capacity for both alignment and capability — it's the training procedure that causes one to degrade the other.

**15. Why does PPO-ptx work better than increasing the KL coefficient?**

Increasing $\beta$ (KL penalty) keeps the policy close to the SFT model in *probability space* — it constrains *how* the model generates text. But the SFT model itself has already drifted from the pretrained model, so staying close to SFT doesn't preserve pretraining capabilities.

PPO-ptx directly optimizes for pretraining performance by including language modeling loss on pretraining data. This explicitly maintains the model's ability to predict text, which is what NLP benchmarks measure. In other words:
- KL penalty says "don't drift from SFT" (but SFT already lost some capabilities)
- PPO-ptx says "also stay good at language modeling" (directly preserving what benchmarks measure)

**16. Could the alignment tax be measured differently?**

Yes — the paper evaluates InstructGPT on benchmarks designed for base language models (zero-shot or few-shot). But InstructGPT is designed to follow instructions, not to complete text in the format these benchmarks expect. If you rephrased benchmark questions as instructions ("Answer the following reading comprehension question: ..."), InstructGPT might actually outperform GPT-3. The alignment tax may partly be a measurement artifact: the model hasn't lost capability, it's just expecting a different input format. Later models like ChatGPT seem to confirm this — they are both aligned and capable when evaluated appropriately.

---

## Evaluation

**17. Is "helpful, honest, harmless" the right framework?**

The HHH framework (from Askell et al. 2021) is useful but incomplete:
- **Conflicts are unavoidable:** A user asks "how do I pick a lock?" — helping a locksmith or enabling a burglar? The model can't always know the user's intent.
- **The paper's approach:** Prioritize helpfulness during training (to get useful behavior), then evaluate harmlessness and truthfulness separately. This means the model learns to be helpful first and is then checked for safety.
- **Alternatives:** Some researchers argue harmlessness should be a hard constraint (never violated), while helpfulness is optimized within that constraint. Others argue for a more nuanced framework that includes fairness, transparency, and privacy.

There's no settled answer — this remains one of the central open questions in AI alignment.

**18. Why didn't RLHF reduce bias?**

Several possible reasons:
- **Bias is subtle:** Labelers were trained to identify toxicity and helpfulness but not subtle stereotypical associations. Bias in Winogender/CrowS-Pairs manifests as statistical preferences (e.g., associating "nurse" with "she"), which labelers wouldn't notice in individual comparisons.
- **Bias is in the pretraining data:** RLHF fine-tunes on top of a model that absorbed biases from internet text. The fine-tuning signal (33k rankings) is tiny compared to pretraining and may not be sufficient to override deeply encoded statistical patterns.
- **The RM doesn't measure bias:** The RM learns what labelers prefer, and labelers weren't rating for bias. To reduce bias, you'd need bias-specific evaluations in the training loop — not just post-hoc benchmarking.

Addressing bias likely requires interventions at multiple levels: pretraining data curation, bias-specific fine-tuning objectives, and evaluation-in-the-loop.

**19. How reliable is the TruthfulQA benchmark?**

TruthfulQA is specifically designed to test common misconceptions and falsehoods that models tend to reproduce (e.g., "What happens if you swallow gum?" where the "wrong" answer is "it stays in your stomach for 7 years"). This is useful for measuring one type of truthfulness — resistance to common myths. But it doesn't capture:
- Factual errors on obscure topics
- Hallucinated citations or statistics
- Subtle inaccuracies in complex explanations
- Whether the model knows when it doesn't know

Good performance on TruthfulQA is encouraging but doesn't mean the model is broadly truthful. It's one narrow slice of truthfulness.

**20. What does a 21% vs 41% hallucination rate actually mean?**

On closed-domain tasks (e.g., summarization where the answer should come from the input), InstructGPT fabricates information 21% of the time vs GPT-3's 41%. This is a meaningful improvement but 21% is still high — roughly 1 in 5 responses on these tasks contains fabricated information. For high-stakes applications (medical, legal, financial), this is far too unreliable. The remaining hallucinations likely occur because:
- The model sometimes "fills in" plausible-sounding details rather than sticking to the source
- RLHF rewards fluent, confident responses, which can incentivize confident hallucination
- The RM may not reliably distinguish fabricated-but-plausible text from factual text

Further reduction likely requires retrieval-augmented generation, better grounding mechanisms, or training the model to cite sources and express uncertainty.

---

## Who Are We Aligning To?

**21. Who decides what "aligned" means?**

The paper is remarkably transparent about this: alignment in InstructGPT means "matching the preferences of ~40 English-speaking contractors guided by OpenAI's instructions." Different labelers from different cultures might:
- Have different norms about directness vs. politeness
- Define "harmful" content differently (e.g., political speech)
- Prioritize different values (individual autonomy vs. community harmony)

"Alignment to labeler preferences" is meaningful as a technical achievement but should not be confused with "alignment to human values" broadly. The paper explicitly states: "We are not claiming that researchers, the labelers we hired, or our API customers are the right source of preferences."

**22. How do OpenAI's labeling instructions influence the outcome?**

Substantially. The labeling instructions define what "good" means — e.g., they instruct labelers to prioritize truthfulness and harmlessness during evaluation. The labelers are executing OpenAI's design choices, not expressing purely independent preferences. The model is therefore aligned to something like "OpenAI's interpretation of good behavior, as implemented by contract workers." This isn't necessarily bad — someone has to make design choices — but it should be understood as a specific set of values, not a universal standard.

**23. Is it possible to align a single model to everyone?**

Probably not in a strong sense. Different cultures, communities, and individuals have genuinely different preferences that can't be averaged into a single "correct" answer. The paper suggests a possible solution: models conditioned on the preferences of specific groups, so different deployments can reflect different values. Another approach is to make models transparent about their constraints and let users understand what values are embedded. Modern systems partially address this through system prompts and customizable behavior, but the fundamental tension remains.

**24. What are the implications of training on API customer prompts?**

The training data comes from early OpenAI API users, who were:
- Selected from a waitlist seeded by OpenAI's networks (tech-heavy, English-speaking)
- Using the API for specific applications (code generation, content writing, Q&A)
- Not representative of the global population who might use language models

This means InstructGPT is optimized for the tasks and communication styles of this particular user group. It may perform differently on tasks or in contexts that were underrepresented in the API data — for example, non-English interactions, culturally specific content, or use cases outside the tech industry.

---

## Connections to Previous Weeks

**25. How does InstructGPT relate to GPT-3's few-shot learning (W1)?**

These findings are complementary, not contradictory:
- GPT-3 (W1) showed that scale improves *capability* — larger models can do more things with few-shot prompting
- InstructGPT (W4) shows that alignment improves *usability* — a properly aligned model is more useful even if smaller

Together they suggest that the ideal model is both large (capable) *and* aligned. The 175B InstructGPT is the best-performing model in the paper, combining GPT-3's scale with RLHF alignment. The 1.3B InstructGPT beating 175B GPT-3 doesn't mean scale doesn't matter — it means alignment matters *more* for user-facing applications.

**26. How does the SFT stage relate to GPT-1's fine-tuning approach (W3)?**

Both use the pretrain-then-fine-tune paradigm, but with key differences:
- **GPT-1 fine-tuning** adapts to specific NLP tasks (classification, entailment) with task-specific heads and structured input formats
- **InstructGPT SFT** adapts to a general "follow instructions" behavior using demonstrations in natural language — no task-specific architecture changes

InstructGPT's SFT is more general: instead of fine-tuning for one task, it fine-tunes for the meta-task of "do what the user asks." It also adds two more stages (RM + PPO) that GPT-1 didn't have.

**27. Could RLHF be applied to GPT-2 or GPT-1?**

In principle, yes — RLHF is model-agnostic. But smaller models have less capacity, so:
- The SFT stage might not produce outputs good enough to meaningfully rank
- The RM might struggle to differentiate between outputs that are all mediocre
- The PPO stage might not have enough "room" to improve — the model's ceiling is lower

The paper's finding that 1.3B InstructGPT beats 175B GPT-3 suggests RLHF can punch above its weight, but there's likely a minimum model size below which the base model simply can't express the desired behaviors. GPT-1 (117M parameters) might be too small; GPT-2 XL (1.5B) would probably work.

---

## Broader Questions

**28. What has changed since this paper was published (2022)?**

Major developments since InstructGPT:
- **ChatGPT (2022):** Applied InstructGPT-like techniques to create a conversational product, proving the approach works at scale for consumer applications
- **Direct Preference Optimization / DPO (2023):** Eliminates the separate RM and PPO stages by directly optimizing the policy on preference data — simpler and often comparably effective
- **Constitutional AI (Anthropic, 2022):** Uses AI-generated feedback guided by a set of principles, reducing reliance on human labelers
- **RLHF at scale:** Modern systems use much larger labeler pools, more sophisticated evaluation criteria, and iterative "red teaming" to find and fix failure modes
- **Unsolved problems:** Hallucination remains a major challenge; bias reduction is still limited; the "who are we aligning to?" question is still open; reward hacking continues to be a concern at scale

**29. Is RLHF the best approach to alignment?**

RLHF is one of several approaches, each with tradeoffs:
- **DPO (Direct Preference Optimization):** Simpler (no RM or PPO), but may be less flexible for complex preference structures
- **Constitutional AI:** Reduces human labor by using AI self-critique, but requires well-designed principles and may amplify model biases
- **Debate:** Two models argue for different answers and a human judges — theoretically scalable but not yet practical at scale
- **Process reward models:** Reward intermediate reasoning steps rather than just final outputs — promising for reducing hallucination
- **RLHF strengths:** Well-tested, scalable, effective in practice. Its main weaknesses are cost (human labelers are expensive), the "who are we aligning to" problem, and reward hacking.

No single approach is clearly "best" — most modern systems combine multiple techniques.

**30. What are the risks of making models better at following instructions?**

The paper itself highlights this concern: when explicitly prompted to be toxic, InstructGPT is *more* toxic than GPT-3 (Figure 39 in the appendix). A model that follows instructions well follows *all* instructions well, including harmful ones. This creates a dual-use problem:
- A helpful model is also a more effective tool for generating misinformation, phishing emails, or harmful content
- The same capability that lets it follow "explain quantum physics simply" also lets it follow "write a convincing scam email"

Mitigations include refusal training (teaching the model to decline harmful requests), output filtering, and use-case restrictions. But the tension is fundamental: instruction-following capability is inherently dual-use. The paper argues that alignment techniques (including refusal) are still net positive, but acknowledges the risk.
