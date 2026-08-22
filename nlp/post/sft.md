---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---
# Supervision

Notations:
- $x$: Prompt (input token seq)
- $y$: Response (output token seq)
- $y_t$: $t$-th response token
- $y_{<t}$: Response tokens before position $t$
- $|y|$: Response length (#tokens)
- $\mathcal{V}$: Vocabulary
- $\mathcal{D}$: Dataset
- $\pi_\theta$: LM policy (next-token distribution)

&nbsp;

## SFT
- **What**: Train a next-token prediction model on curated $(x,y)$ pairs.
- **Why**: Shift probability distribution onto desired behavior.
- **How**: Minimize NLL of response tokens.

```{note} Math
:class: dropdown
Objective:

$$
\mathcal{L}_\text{SFT}(\theta)=-\mathbb{E}_{(x,y)\sim\mathcal{D}}\left[\log\pi_\theta(y|x)\right]=-\mathbb{E}_{(x,y)\sim\mathcal{D}}\left[\sum_{t=1}^{|y|}\log\pi_\theta(y_t|x,y_{<t})\right]
$$
```

```{tip} Derivation
:class: dropdown
*What is SFT actually minimizing?*

1. The dataset defines a conditional $p_\mathcal{D}(y|x)$; the model defines $\pi_\theta(y|x)$.
2. Expand the **forward KL** from data to model:

    $$
    \text{KL}(p_\mathcal{D}\|\pi_\theta)=\mathbb{E}_{y\sim p_\mathcal{D}}[\log p_\mathcal{D}(y|x)]-\mathbb{E}_{y\sim p_\mathcal{D}}[\log\pi_\theta(y|x)]=-H(p_\mathcal{D})+H(p_\mathcal{D},\pi_\theta)
    $$

3. $H(p_\mathcal{D})$ is $\theta$-free → minimize forward KL = minimize CE = MLE.
4. Forward KL is **mode-covering**: wherever $p_\mathcal{D}(y|x)>0$, driving $\pi_\theta(y|x)\to0$ costs $\to\infty$.
5. → The model must place mass on **every** demonstrated response, including mutually contradictory ones.
6. → SFT averages the demonstrations; it cannot prefer among them. That gap is exactly what preference optimization & RL fill.
```

```{attention} Q&A
:class: dropdown
*Pros?*
- Dense signal ← the whole target sequence is supervised.
- Direct control over format, tone, refusal style, tool syntax, etc.

*Cons?*
- **Imitation ceiling**: Cannot exceed the best response in $\mathcal{D}$.
- **Mode-covering**: Contradictory demonstrations get averaged into a blurry compromise.
- Overfitting on small data.

*Why does SFT on facts the model doesn't know induce hallucination?*
- The target is a confident assertion the model has no internal support for.
- → The only generalizable thing to learn is the *behavior* of asserting unsupported facts confidently.
- Empirically: unknown-knowledge examples are fit much slower, and once fit, hallucination rises on **other** questions. {cite:p}`gekhman2024does`
- → Prefer demonstrations of what the model already knows; teach abstention explicitly.

*Why does SFT on the model's own correct outputs beat SFT on stronger off-policy data?*
- An off-policy target can be extremely improbable under the policy → large gradients that move weights far → forgetting & miscalibration.
- On-policy targets sit near the model's own distribution → small, targeted updates.
- → The basis of RFT & on-policy distillation.

*What is exposure bias?*
- Training conditions on the **ground-truth** prefix (teacher forcing); inference conditions on the model's **own** prefix.
- → The model is never trained on the states it reaches after its own mistakes → errors compound over long generations.
- Fix direction: put the model's own samples into training (RFT, on-policy distillation, RL).

*Why not just train more epochs?*
- Post-training sets are $10^3$–$10^6$ samples against $10^9$+ params → memorization is immediate.
- Symptom: train loss ⬇️, held-out win rate flat, generations become templated.
- → 2 epochs is the common default; see [Recipe](#recipe).
```

&nbsp;

### Instruction Tuning
- **What**: SFT on many tasks phrased as natural-language instructions. {cite:p}`wei2021finetuned`
- **Why**: Zero-shot ability does not fall out of single-task fine-tuning.
    - Pretrained LMs are strong few-shot but weak zero-shot ← a bare instruction does not look like pretraining text.
    - Fine-tuning on one task buys that task & nothing else.
- **How**:
    1. Templatize many existing datasets into instruction form, w/ several phrasings each.
    2. Mix them, capped per dataset so no giant set dominates.
    3. SFT.
    4. Evaluate on held-out task **clusters**, ❌held-out examples.

```{attention} Q&A
:class: dropdown
*Why does it generalize to unseen tasks?*
- The learned skill is "read the instruction, then comply", ❌any individual task.
- **Diversity** is the driver: held-out performance rises w/ #task clusters and had not saturated in the original ablation.
- Removing the instructions (same data, no template) destroys the gain → the natural-language framing is load-bearing.

*Why did it make small models worse?*
- FLAN swept 422M / 2B / 8B / 68B / 137B: ⬆️held-out at 68B & 137B, ⬇️held-out at **8B and below**.
- Proposed cause: small capacity is fully consumed learning the ~40 tuning tasks → nothing left for the meta-skill.
- ⚠️ Today's small models are instruction-tuned successfully — data quality & pretraining scale changed, ❌the result was overturned.

*Instruction tuning vs chat SFT?*
- Same objective, different data: many short NLP tasks & single turn vs open-ended multi-turn dialogue.
- Modern mixes contain both, plus code, math, safety, & tool-use traces.

*Why cap each dataset?*
- Source sizes span orders of magnitude → uncapped, one dataset becomes the whole gradient.
- Capping trades raw tokens for task diversity, which is what actually transfers.
```

&nbsp;

### CPT
- **Name**: Continued Pretraining {cite:p}`gururangan2020dont`
- **What**: More next-token training on a raw target-domain corpus.
- **Why**: Some gaps are in the base distribution, ❌in the behavior.
    - Domain jargon, a new language, or a new text modality (code, legal, clinical) may be rare or absent in the original mix.
    - SFT sets are far too small to move what the model knows.
- **How**:
    1. Collect raw domain text, orders of magnitude larger than any SFT set.
    2. **Replay** a slice of the original pretraining mix in every batch.
    3. Re-warm up the LR, then decay again.
    4. Optionally extend tokenizer/context, then SFT as usual on top.

```{attention} Q&A
:class: dropdown
*DAPT vs TAPT?*
- **DAPT** (domain-adaptive) → a large corpus generic to the domain.
- **TAPT** (task-adaptive) → the task's own unlabeled text; tiny but exactly on-distribution.
- Both help & they compose: DAPT → TAPT → SFT.

*Why replay original data?*
- Pure domain text → the model drifts onto it & loses general ability → [catastrophic forgetting](../../dl/issues.md#catastrophic-forgetting).
- Replay keeps the old distribution in the gradient at a fraction of its original cost.

*Why re-warm up the LR?*
- Resuming at the final (tiny) pretraining LR barely moves the weights → CPT does almost nothing.
- Jumping straight to the peak LR spikes the loss & destroys prior capability.
- → Warm up to an intermediate peak, then decay.

*When is CPT the wrong tool?*
- Facts that change → retrieval, ❌weights.
- Format/behavior → SFT; it is far cheaper & more precise.
- → CPT is for a **distribution** shift: vocabulary, style, language, idioms.

*Why extend the tokenizer?*
- A domain the tokenizer never saw is shredded into many sub-word pieces → ⬆️sequence length, ⬇️effective context.
- ⚠️ New embedding rows start random → they need warm-up & enough data, or they stay noise.
```

&nbsp;

### RFT
- **Name**: Rejection sampling Fine-Tuning {cite:p}`yuan2023scaling`
- **What**: SFT on the model's own samples that pass a filter.
- **Why**: Human demonstrations are the bottleneck, & the model already produces good ones some of the time.
    - Checking a response is far cheaper than writing one.
    - Self-generated targets are already on-distribution → easier to fit than expert text.
- **How**:
    1. Sample $k$ responses per prompt from the curr policy at $T>0$.
    2. Keep the ones a verifier / RM accepts.
    3. Dedup — by reasoning path, ❌only by final answer.
    4. Fine-tune the **base** model on the survivors; optionally repeat.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $v(x,y)\in\{0,1\}$: Verifier / filter outcome.
    - $\mathcal{D}^+$: Accepted samples.
- Params:
    - $\pi_{\theta_\text{old}}$: Policy that generated the samples.

Process:

1. $y_i\sim\pi_{\theta_\text{old}}(\cdot|x)$ for $i=1,\dots,k$.
2. $\mathcal{D}^+=\{(x,y_i)\mid v(x,y_i)=1\}$.
3. Minimize $\mathcal{L}_\text{SFT}$ on $\mathcal{D}^+$.

Objective (reward-weighted MLE form):

$$
\mathcal{L}_\text{RFT}(\theta)=-\mathbb{E}_{x\sim\mathcal{D},\ y\sim\pi_{\theta_\text{old}}(\cdot|x)}\left[v(x,y)\log\pi_\theta(y|x)\right]
$$

At the first update after sampling, $\pi_{\theta_\text{old}}=\pi_\theta$, so **if** the samples are drawn at $T=1$ & kept unfiltered by anything other than $v$,

$$
\nabla_\theta\mathcal{L}_\text{RFT}=-\mathbb{E}_{y\sim\pi_\theta}\left[v(x,y)\nabla_\theta\log\pi_\theta(y|x)\right]
$$

which is the REINFORCE gradient w/ $r=v$ & baseline $b=0$. The published recipe breaks both conditions ($T=0.7$, then dedup), so the correspondence is an idealization, ❌an identity.
```

```{attention} Q&A
:class: dropdown
*So is this RL?*
- The gradient coincides w/ REINFORCE only under conditions the recipe does not meet: 1st step after sampling, $T=1$, ❌dedup, binary reward, ❌baseline.
- In practice $T=0.7$, duplicate reasoning paths are removed, & the fine-tune restarts from the **base** model — none of which any policy-gradient method does.
- Everything RL adds is absent: ❌baseline/advantage, ❌importance ratio, ❌KL anchor, ❌fresh rollouts per update.
- → Read it as **one policy-improvement step** (EM-style), ❌an optimization loop.

*Why does it saturate?*
- It can only train on what the policy already produces → the ceiling is pass@$k$ at the sampling budget.
- Prompts the model never solves contribute nothing, ever, no matter how many rounds you run.

*Why dedup?*
- Easy prompts yield many accepted samples → they dominate the filtered set → the mix skews easy.
- Distinct reasoning paths carry information; duplicate final answers do not.

*RFT vs RL, in practice?*
- ✅Stability & infra ← the ordinary SFT trainer, ❌critic, ❌ratio, ❌weight-sync loop.
- ❌Sample efficiency ← every rejected sample is discarded; RL extracts gradient from failures too.
- ✅Safe default when the verifier is reliable but the RL stack is not available.

*What can serve as the filter?*
- Verifier (exact match, unit tests) → precise, verifiable domains only.
- RM / LM judge → broad coverage, hackable → see [reward hacking](../rh.md).
- Human → best & least scalable.

*Why sample at $T>0$?*
- Greedy decoding returns one response per prompt → ❌diversity → the accepted set is tiny.
- Higher $T$ ⬆️coverage of solvable prompts but ⬆️false accepts (right answer, broken reasoning).
```

&nbsp;

#### STaR
- **Name**: Self-Taught Reasoner {cite:p}`zelikman2022star`
- **What**: RFT on rationales, w/ a hinted retry for the failures.
- **Why**: Rationale data does not exist at scale.
    - Final answers are cheap to label; step-by-step reasoning is not.
    - Plain filtering stalls: unsolved problems never produce a rationale, so the training set is permanently the easy subset.
- **How**:
    1. Few-shot prompt the model for rationale + answer.
    2. Keep rationales whose answer is correct.
    3. **Rationalize** the failures: re-prompt w/ the correct answer as a hint, keep the backward rationale it produces.
    4. Fine-tune from the **original** model on everything kept; repeat.

```{attention} Q&A
:class: dropdown
*Why rationalize instead of dropping failures?*
- W/o it the curriculum never advances past what the model can already solve.
- Rationalization manufactures training signal for exactly the problems it cannot yet solve.

*Why is rationalization risky?*
- The rationale is written **knowing** the answer → it can be a post-hoc justification that never actually derives it.
- → Teaches plausible-sounding reasoning that carries no computational weight.

*Why fine-tune from the original model every round?*
- Fine-tuning on top of fine-tunes compounds overfitting to the earlier, easier rounds.
- Restarting keeps the model matched to the current, harder dataset.

*How does this relate to intelligence?*
- The model supplies both the hypotheses & the filter for its own training data; the human supplies only the answer key.
- The limit is sharp: it can bootstrap only what it can already occasionally produce.
- → Self-improvement here is amplification of an existing prior, ❌open-ended discovery.
```

## Practice
### Data
- **What**: The dataset, not the loss, is the method.
- **Why**: The objective is fixed & has no free parameters describing behavior.
    - SFT's optimum **is** the data distribution → whatever is in the mix becomes the model.
    - Two runs w/ identical hyperparameters & different mixes produce unrecognizably different assistants.
- **How**: 4 levers, in descending order of impact.
    1. **Quality**: A few thousand carefully written examples beat a million scraped ones.
    2. **Diversity**: #distinct task types & formats drives generalization more than #samples.
    3. **Decontamination**: n-gram & embedding match against every benchmark you intend to report.
    4. **Dedup**: Near-duplicate prompts silently multiply their own weight.

```{attention} Q&A
:class: dropdown
*How much data?*
- Tone, format, persona → $10^3$ examples.
- General-purpose assistant → $10^5$–$10^6$, heavily mixed.
- A capability the base model lacks → not an SFT problem; go back to CPT or pick a better base.

*Human vs synthetic data?*
- Synthetic generation bootstrapped from a seed set is how most open instruction data was built. {cite:p}`wang2022selfinstruct`
- ✅Cheap, scalable, easy to target a format.
- ❌Inherits the generator's distribution wholesale: its style, its errors, its refusal boundaries, its blind spots.
- → The **filter** is what separates useful synthetic data from noise.

*Why is decontamination not optional?*
- Instruction mixes are scraped or generated from the same sources benchmarks come from.
- Leakage is invisible in the training curves & inflates exactly the number you report.
- ⚠️ Synthetic data generated by a model that memorized the benchmark leaks it too — n-gram matching against the *seed* set is not enough.

*What does a bad mix look like?*
- 1 dominant source → every answer in that source's voice.
- Uniformly long responses → verbosity; uniformly short → truncated reasoning.
- ❌Refusals → complies w/ anything; ⬆️refusals → refuses the benign.
- ❌Multi-turn examples → the model is coherent for exactly 1 turn.

*Why does quality beat quantity so sharply here?*
- Pretraining already supplied the capability; SFT selects a **style of response**.
- A style is a low-complexity target → few, consistent examples define it precisely.
- Contradictory examples are worse than no examples ← forward KL forces the model to cover both.
```

&nbsp;

### Recipe
- **What**: Default hyperparameters for supervised post-training.
- **Why**: Pretraining intuitions invert at this scale.
    - The dataset is $10^{-4}$ the size, and the model is already at a good optimum.
    - → More steps & bigger LR, which help in pretraining, actively destroy the checkpoint here.
- **How**: Anchor on a published, fully specified recipe & sweep from there.

````{dropdown} Table: Tülu 3 Hyperparameters {cite:p}`lambert2024tulu`
| Hyperparameter | SFT 8B | SFT 70B | DPO 8B | DPO 70B |
|:--|:--|:--|:--|:--|
| LR | $5\times10^{-6}$ | $2\times10^{-6}$ | $5\times10^{-7}$ | $2\times10^{-7}$ |
| LR schedule | Linear | Linear | Linear | Linear |
| Effective batch size | 128 | 128 | 128 | 128 |
| Max token length | 4,096 | 4,096 | 2,048 | 2,048 |
| Warmup ratio | 0.03 | 0.03 | 0.1 | 0.1 |
| Epochs | 2 | 2 | 1 | 1 |
| $\beta$ | — | — | 5 | 5 |

SFT uses **sum** loss; DPO is **length-normalized**, which is why $\beta=5$ rather than the usual $0.1$.
````

```{attention} Q&A
:class: dropdown
*Why is the preference-stage LR ~10× below the SFT LR?*
- The preference loss has a degenerate direction (grow the margin forever) that SFT's does not.
- Large steps reach it immediately → both log-probs collapse while reward accuracy stalls.

*Why 1 epoch for preference optimization & 2 for SFT?*
- Preference pairs are memorized almost instantly ← the label is 1 bit and the pairs are few.
- A 2nd epoch reliably ⬆️margin & ⬇️held-out win rate.

*What should be monitored?*
- SFT: held-out **loss** + actual generations. Train loss alone hides both overfitting & format collapse.
- Preference: reward **accuracy**, both reward levels separately, & KL to $\pi_\text{ref}$.
- Always: mean generation length. Silent verbosity growth is the most common regression.

*LoRA vs full FT LR?*
- LoRA wants ~10× more (order $10^{-4}$) ← the branch starts at 0 & has few params.
- Reusing the full-FT LR is the single most common cause of "LoRA did nothing".

*Why sweep the LR before anything else?*
- Reported method rankings routinely flip under per-method LR tuning.
- → A baseline at someone else's LR is not a baseline.
```

&nbsp;

### Alignment Tax
- **What**: Capability regression on unrelated tasks caused by alignment training. {cite:p}`ouyang2022training`
- **Why**: The objective mentions only the alignment data.
    - Everything not in $\mathcal{D}$ is unconstrained → free to degrade at no cost to the loss.
    - Narrow data + full-parameter updates → the model rewrites circuitry shared w/ untouched capabilities.
    - InstructGPT measured it directly: regressions on SQuAD, DROP, HellaSwag, & WMT translation.
- **How**: 4 mitigations, in increasing order of bluntness.
    1. **Mix in pretraining data** — InstructGPT's PPO-ptx greatly reduced the regressions w/o costing labeler preference.
    2. **Anchor to $\pi_\text{ref}$** — a KL term bounds how far the policy may move.
    3. **PEFT** — the frozen base caps the achievable change structurally.
    4. **Model merging** — average the base & aligned checkpoints after the fact.

```{attention} Q&A
:class: dropdown
*Is the tax inevitable?*
- ❌ It is a fit-vs-regularization trade-off, ❌a law. Data mixing largely closed it in InstructGPT.
- ✅ But it is invisible unless measured — nothing in the training loss reports it.

*Why does it get worse w/ more alignment?*
- Longer training → further from $\pi_\text{ref}$ → more of the shared representation is repurposed.
- The alignment metric keeps improving throughout, so the training signal never objects.

*How should it be measured?*
- Fix a capability suite (knowledge, math, code, long-context) **before** training.
- Run it on the base & on every aligned checkpoint. Report both numbers, always.

*How does it relate to catastrophic forgetting?*
- Same mechanism, different framing: [catastrophic forgetting](../../dl/issues.md#catastrophic-forgetting) is the general phenomenon.
- "Alignment tax" names the case where the new task is *alignment* & the lost task is *general capability*.
```

&nbsp;

### Design Space
- **What**: The 4 decisions that define any supervised post-training run.
- **Why**: The methods differ by small deltas, so the deltas are the whole comparison.
- **How**:
    1. **Data source**: Human / teacher / self-generated — plus the filter applied to it.
    2. **Objective**: NLL (SFT) / divergence to a teacher (KD) / preference loss.
    3. **Parameterization**: Full / LoRA-family / prompt-family.
    4. **Anchor**: Reference KL / an SFT term / length normalization / none.

````{dropdown} Table: Methods at a Glance
| Method | Data | Target | Anchor | Needs | Delta |
|:--|:--|:--|:--|:--|:--|
| SFT | Human $(x,y)$ | Token one-hot | — | Demonstrations | The baseline |
| Instruction Tuning | Many tasks, templated | Token one-hot | — | Task diversity | Zero-shot generalization |
| CPT | Raw domain text | Token one-hot | Replay mix | A large corpus | Moves the base distribution |
| RFT | Self-generated, filtered | Token one-hot | — | Verifier | On-policy targets, ❌humans |
| Sequence-Level KD | Teacher samples | Token one-hot | — | Teacher API | Black-box transfer |
| KD | Fixed corpus | Teacher distribution | — | Teacher logits | Dense per-token signal |
| GKD | Student samples | Teacher distribution | — | Teacher + generation | ❌Exposure bias |
| DPO | Pairs | Preference label | $\pi_\text{ref}$ KL | Preference data | ❌RM, ❌rollouts |
| IPO | Pairs | Preference label | $\pi_\text{ref}$ KL | Preference data | Bounded margin |
| KTO | Unpaired labels | Binary label | $\pi_\text{ref}$ KL | Thumbs up/down | ❌Pairing |
| ORPO | Pairs | Preference + NLL | SFT term | Preference data | 1 stage |
| SimPO | Pairs | Preference label | Length norm | Preference data | ❌Reference model |
````

```{attention} Q&A
:class: dropdown
*What should I actually run?*
- Base model → assistant: SFT on a curated mix, then DPO on preference pairs. This is the default & it works.
- Small model, strong teacher available: sequence-level KD (+ a filter) beats writing demonstrations.
- Verifiable domain, no RL stack: RFT.
- Feedback arrives as thumbs up/down: KTO.
- 1 GPU: QLoRA.
- Verifiable domain **and** an RL stack: stop here & go to [RL for LLMs](rl.md).

*What has the field converged on?*
- **SFT → preference optimization** as the default 2-stage recipe.
- **LoRA** as the default parameterization below a full-FT budget.
- **Length normalization** somewhere in the preference objective — nearly universal after the verbosity problem became undeniable.
- **On-policy preference pairs** over off-policy ones.

*What is still contested?*
- Reference-free (ORPO, SimPO) vs reference-anchored (DPO, IPO, KTO) — no consensus, & results flip w/ per-method LR tuning.
- DPO-family vs online RL — the structural trade-off is clear (coverage vs cost), the empirical ranking is not, and it depends heavily on RM quality & data.
- Whether to mask the prompt, and how to normalize the loss.

*What is unsolved?*
- **The imitation ceiling.** Every objective here fits a target that already exists. Nothing discovers a better response than the one it was handed.
- **Predicting data quality.** "Curate high-quality data" is the dominant lever & there is no reliable a-priori metric for it.
- **The 1-bit channel.** A preference label carries 1 bit about an entire pair; the *reason* for the preference is never recorded.
- **Forgetting.** Every method trades general capability for the target behavior, & the exchange rate is unmeasured until you look.

*What does this say about intelligence?*
- Every method here **redistributes** probability mass over behaviors the pretrained model can already emit. None expands the hypothesis space.
- The interesting boundary is RFT & GKD: the model generates its own training data, so improvement is bounded by its own sampling distribution — self-improvement that is real but strictly self-limited.
- → Elicitation is remarkably powerful, and remarkably not the same thing as learning something new.
```

&nbsp;
