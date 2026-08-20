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
# Supervised Post-Training
Everything that turns a pretrained LM into an assistant using a **fixed dataset** & a **differentiable loss**: SFT, distillation, offline preference optimization, and the parameter-efficient ways to run all three.

Policy-gradient methods live in [RL for LLMs](rl.md).

```{dropdown} Table: Shared Notations
| Notation | Meaning |
|:--|:--|
| $x$ | Prompt (input token seq) |
| $y$ | Response (output token seq) |
| $y_t$ | $t$-th response token |
| $y_{<t}$ | Response tokens before position $t$ |
| $\|y\|$ | Response length (#tokens) |
| $\mathcal{V}$ | Vocabulary |
| $\mathcal{D}$ | Post-training dataset |
| $\pi_\theta$ | Policy being trained (the LM) |
| $\pi_\text{ref}$ | Reference policy (frozen, usually the SFT checkpoint) |
| $y_w,y_l$ | Preferred / dispreferred response |
| $\hat{r}_\theta$ | Implicit reward (a log-ratio, ❌a learned RM) |
| $\beta$ | Reward scale / KL coeff |
| $\sigma$ | Logistic sigmoid |
| $W_0$ | Frozen pretrained weight matrix |
| $r$ | Adapter rank |

$r$ overrides the reward symbol used on the RL page — nothing here is trained against a reward.
```

&nbsp;

## Setup
### Pipeline
- **What**: Ordered stages, each supplying a signal the previous one cannot express.
- **Why**: A pretrained LM completes text; it does not answer.
    - Next-token on web text → the likeliest continuation of a question is often another question.
    - Knowledge sits in the weights, but *producing it on demand* is a behavior, ❌a fact.
    - Format, refusal, tone, tool syntax, stopping — none of it is a property of the corpus.
- **How**: 4 stages, ordered by how far past the given data each can reach.
    1. **CPT**: Raw domain text → move the base distribution.
    2. **SFT**: $(x,y)$ demonstrations → install the response behavior.
    3. **Preference optimization**: $(x,y_w,y_l)$ → rank behaviors that demonstrations cannot show.
    4. **RL**: Sampled responses + grader → optimize past every demonstration.

```{dropdown} Table: Stages
| Stage | Data | Signal / sample | Past the data? | Cost driver |
|:--|:--|:--|:--|:--|
| Pretrain | Raw web text | Next token | ❌ | Corpus scale |
| CPT | Raw domain text | Next token | ❌ | Corpus scale |
| SFT | $(x,y)$ | Full target seq | ❌ Imitation ceiling | Human writing |
| Distillation | $(x,y)$ + teacher probs | Full target distribution | ❌ Teacher ceiling | Teacher inference |
| Preference opt | $(x,y_w,y_l)$ | 1 bit | ✅ Weakly (ranks unseen pairs) | Human comparison |
| RL | $x$ + grader | 1 scalar / rollout | ✅ | Rollout throughput |
```

```{attention} Q&A
:class: dropdown
*Why is the order fixed?*
- Preference optimization & RL both **reweight** what the policy already emits → they need a policy that emits well-formed responses.
- Both need $\pi_\text{ref}$ as an anchor → it has to be produced by SFT first.
- Skipping SFT → nearly every response is malformed → uniformly bad grades → ❌signal.

*Which of these are actually supervised?*
- Fixed target + plain backprop, ❌reward: CPT, SFT, distillation, offline preference optimization.
- RFT & on-policy distillation **sample** from the policy but still fit a fixed target → supervised loss on self-generated inputs.
- → The line is not "does it sample?" but "is the loss weighted by a reward?".

*Do you need every stage?*
- SFT alone → usable assistant, poor at trade-offs it was never shown.
- SFT + preference optimization → the standard open-weights recipe.
- +RL → worth it only w/ a reliable grader (verifiable domain) or a trusted RM.

*Is post-training adding capability or exposing it?*
- **Superficial alignment hypothesis** (LIMA): nearly all knowledge & capability come from pretraining; post-training only selects a response distribution. {cite:p}`zhou2023lima`
- ✅Evidence: ~1k curated examples produce a competitive chat model.
- ❌Evidence: distillation on reasoning traces raises accuracy on *unseen* problems, ❌only formatting.
- → Contested. Safest reading: SFT ≈ elicitation & format, RL ≈ sharpening, genuinely new knowledge ≈ pretraining/CPT.
```

&nbsp;

### Chat Template
- **What**: Role-tagged serialization of a conversation into one token stream.
- **Why**: The LM sees a flat sequence, ❌structured messages.
    - Nothing in raw text marks where the user's turn ends & the model's begins.
    - W/o a learned boundary, text inside a user turn is indistinguishable from an instruction → injection.
    - No token means "done" → generation never terminates.
- **How**:
    1. Add **special tokens** for turn boundaries & role headers to the vocab.
    2. Serialize messages in order; each turn = header + content + end token.
    3. Train so the end-of-turn token is predicted after every assistant turn.
    4. At inference, append the assistant header (**generation prompt**) & decode until the end-of-turn token.

````{important} Code
:class: dropdown
```python
SPECIALS = {"bot": "<|im_start|>", "eot": "<|im_end|>"}  ## in the vocab, never spellable by users

def render(messages, add_generation_prompt=True):
    ## messages: [{"role": "user"|"assistant"|"system", "content": str}]
    out = []
    for m in messages:
        ## header + content + explicit terminator -> the boundary is a TOKEN, not whitespace
        out.append(f"{SPECIALS['bot']}{m['role']}\n{m['content']}{SPECIALS['eot']}\n")
    if add_generation_prompt:
        ## inference-only: open the assistant turn and stop, so the model completes it
        out.append(f"{SPECIALS['bot']}assistant\n")
    return "".join(out)

## Example
msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
print(repr(render(msgs[:1])))
## '<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n'
print(repr(render(msgs, add_generation_prompt=False)))
## '<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\nhello<|im_end|>\n'
```
````

```{attention} Q&A
:class: dropdown
*What breaks if the training template ≠ the inference template?*
- One extra/missing token puts every prompt slightly off-distribution.
- Fails silently: the model still answers, just worse — ❌error, ❌crash.
- → The template ships **with** the weights; it is part of the model, ❌of the serving code.

*Why must the markers be special tokens instead of literal strings?*
- Special tokens are inserted by the renderer & excluded when tokenizing user content.
- → A user who types `<|im_end|>` gets ordinary sub-word tokens, ❌the control token.
- Literal-string markers are forgeable → the user can open a fake assistant turn.

*Why train on the end-of-turn token?*
- It is the only token that means "stop" → masked out of the loss, the model never learns to emit it.
- Symptom: correct answer, followed by an invented next user turn, forever.

*Base vs instruct checkpoint?*
- Base: ❌template, ❌special tokens → prompt w/ few-shot text, expect continuation.
- Instruct: template is mandatory; prompting it as raw text is off-distribution.

*Where does the system prompt go?*
- Its own leading turn → the model learns it outranks later user turns.
- Vary it during SFT, otherwise the behavior binds to one exact string.
```

&nbsp;

### Loss Masking
- **What**: CE computed on response tokens only.
- **Why**: Prompt tokens are given, never generated.
    - Scoring them trains the model to *produce* user turns → capacity spent on the wrong distribution.
    - Multi-turn: every assistant turn is a target, every user turn is context.
- **How**:
    1. Render the conversation; record the span of each assistant turn.
    2. Set labels to the ignore index everywhere else.
    3. Shift labels by 1 (position $t$ predicts token $t+1$).
    4. Aggregate the surviving token losses.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathcal{B}$: Minibatch of $(x,y)$ pairs.
    - $\mathcal{M}$: Set of unmasked (trainable) token positions.
- Misc:
    - $\mathbb{1}[\cdot]$: Indicator.
    - $Z$: Normalizer.

Objective:

$$
\mathcal{L}(\theta)=-\frac{1}{Z}\sum_{(x,y)\in\mathcal{B}}\sum_{t=1}^{|y|}\mathbb{1}[t\in\mathcal{M}]\log\pi_\theta(y_t|x,y_{<t})
$$

$Z$ is the entire design choice:

$$
Z=\begin{cases}1 & \text{sum loss}\\ |\mathcal{B}| & \text{sample mean}\\ \sum_{(x,y)\in\mathcal{B}}|y| & \text{token mean}\end{cases}
$$
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn.functional as F

IGNORE = -100  ## torch's default ignore_index

def build_labels(input_ids, spans):
    ## spans: [(start, end)] half-open token ranges of the ASSISTANT turns
    labels = torch.full_like(input_ids, IGNORE)
    for s, e in spans:
        labels[s:e] = input_ids[s:e]
    return labels

def sft_loss(logits, labels, reduction="token_mean"):
    ## next-token shift: position t predicts token t+1
    logits, labels = logits[:, :-1], labels[:, 1:]
    tok = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
        ignore_index=IGNORE, reduction="none",
    ).view(labels.shape)                        ## (B, T-1), 0 at ignored positions
    keep = labels != IGNORE
    if reduction == "token_mean":
        return tok.sum() / keep.sum()           ## every TOKEN weighted equally
    if reduction == "sample_mean":
        return (tok.sum(-1) / keep.sum(-1)).mean()  ## every SAMPLE weighted equally
    return tok.sum()                            ## sum loss: no denominator

## Example
ids = torch.randint(0, 50, (1, 8))
labels = build_labels(ids[0], [(4, 8)]).unsqueeze(0)  ## first 4 tokens are the prompt
print(labels)                                         ## [-100 x4, then real ids]
print(sft_loss(torch.randn(1, 8, 50), labels).item())
```
````

```{attention} Q&A
:class: dropdown
*Is masking the prompt always right?*
- Standard, ❌universal. Scoring the prompt is a mild regularizer on tiny datasets & harmless when prompts are in-domain text.
- ❌ For long prompts w/ short answers: the loss becomes mostly prompt → the answer signal is drowned.

*Sample mean vs token mean vs sum?*
- Sample mean → each response contributes equally → per-token weight $\propto\frac{1}{|y|}$ → long responses down-weighted.
- Token mean → each token equal → long responses dominate the batch.
- Sum → each token equal, and the gradient magnitude scales w/ the batch's token count → needs a re-tuned LR.
- Tülu 3 found sum loss beat mean loss under a matched LR sweep. {cite:p}`lambert2024tulu`
- ⚠️ W/ gradient accumulation, a token mean taken **per micro-batch** is not the token mean of the full batch.

*Why is the ignore index $-100$ and not $0$?*
- $0$ is a valid token id → it would silently train on whatever token $0$ is.
- $-100$ is torch's sentinel: no gradient, and the position is excluded from the denominator.

*Multi-turn: train on the last turn only, or all of them?*
- All assistant turns → more signal per forward pass.
- Last only → correct when earlier assistant turns came from a *different* model; otherwise wasted compute.
```

&nbsp;

### Packing
- **What**: Concatenating short samples into full-length sequences.
- **Why**: Padding is compute spent on nothing.
    - Instruction data is short & highly length-varied → a padded batch can be mostly pad.
    - Attention is quadratic in sequence length → padding to the batch's longest sample is doubly wasteful.
- **How**:
    1. Concatenate rendered samples until the context window is full.
    2. Reset position ids at every sample boundary.
    3. Block cross-sample attention (block-diagonal mask, or a varlen attention kernel w/ cumulative lengths).
    4. Apply prompt masking as usual.

```{attention} Q&A
:class: dropdown
*What happens w/o cross-sample attention blocking?*
- Tokens attend to unrelated preceding samples → **contamination**.
- Tolerable in pretraining (long docs, few boundaries), harmful in SFT (short samples → many boundaries per sequence).
- Symptom: answers that leak the topic of the previous sample in the pack.

*Why reset position ids?*
- Otherwise the 5th sample in a pack is only ever trained at positions 3000+.
- → At inference every prompt starts at position 0 → off-distribution.

*Packing vs length bucketing?*
- Bucketing (sort by length, batch similar) → ❌boundary logic, but ⬇️batch diversity & still pads.
- Packing → ~100% token utilization, needs the mask plumbing.

*Does packing change the objective?*
- Token mean → yes: per-sequence token counts change, so per-sample weights shift.
- Sum loss → no.

*Does truncation matter?*
- Splitting a sample across two packs teaches the model to stop mid-answer & to start mid-sentence.
- → Drop over-length samples, or keep each sample whole (best-fit packing).
```

&nbsp;

## SFT
- **Name**: Supervised Fine-Tuning
- **What**: Next-token CE on curated $(x,y)$ pairs.
- **Why**: Behavior must be **shown**, ❌described.
    - Prompting alone → format is unreliable, & the instructions burn context on every call.
    - The target is a *distribution over responses*; the only cheap handle on a distribution is samples from it.
- **How**:
    1. Collect $(x,y)$ pairs — human-written, distilled, or filtered self-generated.
    2. Render w/ the chat template; mask the prompt tokens.
    3. Minimize the NLL of the response tokens under teacher forcing.
    4. 1–3 epochs at an LR 1–2 orders below pretraining.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $(x,y)\sim\mathcal{D}$: Prompt-response pair.
- Params:
    - $\theta$: LM params.

Sequence factorization:

$$
\log\pi_\theta(y|x)=\sum_{t=1}^{|y|}\log\pi_\theta(y_t|x,y_{<t})
$$

Objective:

$$
\mathcal{L}_\text{SFT}(\theta)=-\mathbb{E}_{(x,y)\sim\mathcal{D}}\left[\log\pi_\theta(y|x)\right]=-\mathbb{E}_{(x,y)\sim\mathcal{D}}\left[\sum_{t=1}^{|y|}\log\pi_\theta(y_t|x,y_{<t})\right]
$$
```

```{tip} Derivation
:class: dropdown
*What is SFT actually minimizing, and what does that imply?*

1. The dataset defines a conditional $p_\mathcal{D}(y|x)$; the model defines $\pi_\theta(y|x)$.
2. Expand the **forward** KL from data to model:

    $$
    \text{KL}(p_\mathcal{D}\|\pi_\theta)=\mathbb{E}_{y\sim p_\mathcal{D}}[\log p_\mathcal{D}(y|x)]-\mathbb{E}_{y\sim p_\mathcal{D}}[\log\pi_\theta(y|x)]=-H(p_\mathcal{D})+H(p_\mathcal{D},\pi_\theta)
    $$

3. $H(p_\mathcal{D})$ is $\theta$-free → minimizing forward KL $\equiv$ minimizing CE $\equiv$ MLE.
4. Forward KL is **mode-covering**: wherever $p_\mathcal{D}(y|x)>0$, driving $\pi_\theta(y|x)\to0$ costs $\to\infty$.
5. → The model must place mass on **every** demonstrated response, including mutually contradictory ones.
6. → SFT averages the demonstrations; it cannot prefer among them. That gap is exactly what preference optimization & RL fill.
```

```{attention} Q&A
:class: dropdown
*Pros?*
- Stable & cheap ← 1 model in memory, ❌sampling loop, ❌reward model.
- Dense signal ← the whole target sequence is supervised, ❌1 scalar per response.
- Direct control over format, tone, refusal style, tool syntax.

*Cons?*
- **Imitation ceiling** ← cannot exceed the best response in $\mathcal{D}$.
- ❌Negative signal ← the loss only pushes probability **up**; nothing says what not to do.
- Mode-covering → contradictory demonstrations get averaged into a blurry compromise.
- Overfits fast on small data → memorized phrasings, ⬇️output diversity.

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
    - SFT sets are far too small to move what the model *knows*.
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
    4. SFT on the survivors; optionally repeat.

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

At the first update after sampling, $\pi_{\theta_\text{old}}=\pi_\theta$, so

$$
\nabla_\theta\mathcal{L}_\text{RFT}=-\mathbb{E}_{y\sim\pi_\theta}\left[v(x,y)\nabla_\theta\log\pi_\theta(y|x)\right]
$$

which is the REINFORCE gradient w/ $r=v$ & baseline $b=0$.
```

```{attention} Q&A
:class: dropdown
*So is this RL?*
- The gradient coincides w/ REINFORCE **only** at the first step after sampling, w/ a binary reward and no baseline.
- Everything RL adds is absent: ❌baseline/advantage, ❌importance ratio, ❌KL anchor, ❌fresh rollouts per update.
- In practice it is run as plain SFT over a frozen filtered set for multiple epochs → silently, uncorrectedly off-policy.
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

&nbsp;

## Distillation
- **What**: Training a student to match a teacher's output distribution. {cite:p}`hinton2015distilling`
- **Why**: The teacher's behavior is wanted at the student's cost.
    - Serving a frontier model is expensive, & capability per param is not fixed.
    - A one-hot label carries at most $\log_2|\mathcal{V}|\approx17$ bits per position; the teacher's full distribution carries the relative probabilities of every **wrong** token too.
    - Those relatives encode the teacher's similarity structure ("dark knowledge") — the part a hard label throws away.
- **How**:
    1. Choose the token set to distill on: fixed corpus / teacher samples / student samples.
    2. Teacher forward pass → a next-token distribution at every position.
    3. Minimize a divergence between student & teacher distributions, optionally mixed w/ hard-label CE.
    4. Soften both sides w/ temperature $\tau$ during training.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $p_T(\cdot|x,y_{<t})$: Teacher next-token distribution.
    - $p_S^\theta(\cdot|x,y_{<t})$: Student next-token distribution.
- Params:
    - $z$: Student logits.
- Hyperparams:
    - $\tau$: Temperature.
    - $\alpha$: Weight on the soft (teacher) term.

Objective:

$$
\mathcal{L}_\text{KD}(\theta)=\mathbb{E}_{(x,y)\sim\mathcal{D}}\left[\frac{1}{|y|}\sum_{t=1}^{|y|}\text{KL}\left(p_T(\cdot|x,y_{<t})\ \|\ p_S^\theta(\cdot|x,y_{<t})\right)\right]
$$

W/ temperature & hard labels:

$$
\mathcal{L}=\alpha\tau^2\,\text{KL}\left(p_T^\tau\ \|\ p_S^{\theta,\tau}\right)+(1-\alpha)\,\text{CE}(y,p_S^\theta)
$$
- $p^\tau$: Softmax over logits divided by $\tau$.
- $\tau^2$: Rescaling — softened gradients shrink as $\tau^{-2}$, so this keeps the two terms comparable.

Gradient w.r.t. student logits (at $\tau=1$):

$$
\nabla_z\,\text{KL}(p_T\|p_S^\theta)=p_S^\theta-p_T,\qquad \nabla_z\,\text{CE}(y,p_S^\theta)=p_S^\theta-\mathbf{e}_{y_t}
$$
- $\mathbf{e}_{y_t}$: One-hot at the target token.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn.functional as F

def kd_loss(student_logits, teacher_logits, labels, tau=2.0, alpha=0.9, ignore=-100):
    ## soften BOTH sides with the same temperature
    s_log = F.log_softmax(student_logits / tau, dim=-1)
    t_prob = F.softmax(teacher_logits / tau, dim=-1)
    ## KL(teacher || student), summed over vocab, averaged over live positions
    kl = (t_prob * (t_prob.clamp_min(1e-9).log() - s_log)).sum(-1)   ## (B, T)
    keep = labels != ignore
    soft = (kl * keep).sum() / keep.sum()
    hard = F.cross_entropy(
        student_logits.reshape(-1, student_logits.size(-1)),
        labels.reshape(-1), ignore_index=ignore,
    )
    ## tau^2 restores the gradient scale the softmax division removed
    return alpha * tau**2 * soft + (1 - alpha) * hard

## Example
B, T, V = 2, 3, 10
labels = torch.randint(0, V, (B, T))
print(kd_loss(torch.randn(B, T, V), torch.randn(B, T, V), labels).item())
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- ⬇️Serving cost at a given capability level.
- ⬆️Sample efficiency ← every position supplies $|\mathcal{V}|$ numbers instead of 1 index.
- Transfers behavior the student would not reach from the same data alone.

*Cons?*
- Teacher inference over the whole corpus → expensive, & it must be redone for new data.
- Ceiling is essentially the teacher.
- Token-level KD needs **logits** & a shared tokenizer → white-box only.
- Distilling a commercial API usually violates its terms.

*Why is matching the distribution better than matching the argmax?*
- CE gradient is $p_S-\mathbf{e}_{y_t}$: informative at 1 vocab entry.
- KD gradient is $p_S-p_T$: a target at **every** vocab entry, including which wrong answers are near-misses.
- → Effectively a much denser label per token.

*Forward vs reverse KL?*
- Forward $\text{KL}(p_T\|p_S)$ → **mode-covering**: the student must cover everything the teacher does, hedging where it lacks capacity.
- Reverse $\text{KL}(p_S\|p_T)$ → **mode-seeking**: the student concentrates on a subset of teacher modes, sharper & more fluent, less diverse.
- → Capacity gap is the deciding factor: the wider it is, the more reverse KL is preferred.

*Can the student ever beat the teacher?*
- ✅ In restricted senses: distilling an expensive procedure (CoT, search, ensembling) into one forward pass, or distilling a *filtered* teacher.
- ✅ **Weak-to-strong**: a strong student supervised by a much weaker teacher recovers a large fraction of the gap — supervision **elicits** latent capability rather than transferring it. {cite:p}`burns2023weak`
- ❌ In general: matching a distribution cannot exceed it.

*Different tokenizers?*
- Token-level KD is undefined ← the vocab index spaces don't correspond.
- → Fall back to sequence-level KD, or align tokenizations approximately.
```

&nbsp;

### Sequence-Level KD
- **What**: SFT on teacher-generated sequences. {cite:p}`kim2016sequence`
- **Why**: Token-level KD needs logits & a fixed target corpus, and matches the wrong object.
    - Frontier teachers expose text, ❌logits.
    - What matters at inference is the distribution over **sequences**, not per-position marginals conditioned on a ground-truth prefix.
- **How**:
    1. Generate responses from the teacher for each prompt (sample, or take the beam-search mode).
    2. Optionally filter w/ a verifier.
    3. Plain SFT on (prompt, teacher response).

```{attention} Q&A
:class: dropdown
*Why does it work w/o any logits?*
- Sampling $y\sim p_T$ then doing MLE minimizes $\text{KL}(p_T\|p_S)$ at the **sequence** level, Monte-Carlo estimated.
- The original formulation instead approximates the teacher's sequence distribution by a point mass at its **mode** (beam output) → SFT on that single output.
- → Either way the teacher's distribution enters through its samples, ❌its probabilities.

*Trade-off vs token-level KD?*
- ✅Black-box, tokenizer-agnostic, reuses the SFT trainer, teacher runs once per prompt.
- ❌ ~$\log_2|\mathcal{V}|$ bits per position instead of a full distribution → far weaker signal per token.

*What is the failure mode?*
- It is SFT → exposure bias returns in full: the student only ever sees teacher trajectories, which may be very improbable under itself.
- Sharpening on the teacher's mode also collapses diversity relative to the teacher.

*Is this just "SFT on synthetic data"?*
- Mechanically identical. Most open "distilled" models are exactly this.
- The distinction that matters is the **filter**: unfiltered teacher output inherits every teacher error.
```

&nbsp;

### GKD
- **Name**: Generalized Knowledge Distillation {cite:p}`agarwal2023onpolicy`
- **What**: Distillation on the **student's own** samples.
- **Why**: Fixed teacher data leaves the student unsupervised exactly where it operates.
    - Trained on teacher trajectories, evaluated on its own → the states it actually visits were never labeled.
    - A low-capacity student cannot match the teacher everywhere, so forward KL on teacher data spends its capacity hedging modes it can never represent.
- **How**:
    1. Sample outputs from the **student**.
    2. Teacher scores those exact tokens → target distributions.
    3. Minimize a divergence at every position.
    4. Mix student-generated & fixed data w/ $\lambda$; pick the divergence via JSD($\beta$).

```{note} Math
:class: dropdown
Notations:
- IO:
    - $(X,Y)$: Fixed dataset of prompts & target responses.
    - $p_S(\cdot|x)$: Student sampling distribution (samples treated as data, no gradient through sampling).
- Hyperparams:
    - $\lambda\in[0,1]$: Fraction of student-generated data.
    - $\beta\in[0,1]$: Divergence interpolation.
- Misc:
    - $\mathcal{D}$: Chosen divergence.

Divergence:

$$
\mathcal{D}_{\text{JSD}(\beta)}(P\|Q)=\beta\,\text{KL}\left(P\ \|\ \beta P+(1-\beta)Q\right)+(1-\beta)\,\text{KL}\left(Q\ \|\ \beta P+(1-\beta)Q\right)
$$
- $\beta\to0$ → recovers forward $\text{KL}(P\|Q)$ (up to scale); $\beta\to1$ → reverse $\text{KL}(Q\|P)$; $\beta=0.5$ → standard JSD.

Objective:

$$
\mathcal{L}_\text{GKD}(\theta)=(1-\lambda)\,\mathbb{E}_{(x,y)\sim(X,Y)}\left[\mathcal{D}(p_T\|p_S^\theta)(y|x)\right]+\lambda\,\mathbb{E}_{x\sim X}\,\mathbb{E}_{y\sim p_S(\cdot|x)}\left[\mathcal{D}(p_T\|p_S^\theta)(y|x)\right]
$$
- $\mathcal{D}(p_T\|p_S^\theta)(y|x)$: Token-averaged divergence over the positions of $y$.
- $\lambda=0$ → supervised KD; $\lambda=1$ → fully on-policy.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn.functional as F

def jsd_beta(t_logits, s_logits, beta=0.5):
    ## generalized JSD: beta -> 0 gives forward KL, beta -> 1 gives reverse KL
    logp_t, logp_s = F.log_softmax(t_logits, -1), F.log_softmax(s_logits, -1)
    ## mixture in log-space for numerical stability
    log_mix = torch.logsumexp(
        torch.stack([logp_t + torch.log(torch.tensor(beta).clamp_min(1e-9)),
                     logp_s + torch.log(torch.tensor(1 - beta).clamp_min(1e-9))]), dim=0)
    kl_t = (logp_t.exp() * (logp_t - log_mix)).sum(-1)
    kl_s = (logp_s.exp() * (logp_s - log_mix)).sum(-1)
    return beta * kl_t + (1 - beta) * kl_s            ## (B, T)

def gkd_loss(teacher, student, fixed_batch, student_samples, beta=0.5, lam=0.5):
    ## student_samples: token ids the STUDENT just generated -> detached, they are DATA
    on = jsd_beta(teacher(student_samples), student(student_samples), beta).mean()
    off = jsd_beta(teacher(fixed_batch), student(fixed_batch), beta).mean()
    return (1 - lam) * off + lam * on

## Example
B, T, V = 2, 4, 8
print(jsd_beta(torch.randn(B, T, V), torch.randn(B, T, V)).shape)  ## torch.Size([2, 4])
```
````

```{attention} Q&A
:class: dropdown
*Why is this not RL, given that it samples from the policy?*
- ❌Reward, ❌advantage, ❌policy-gradient estimator.
- The samples are **detached** & used as input positions; the gradient flows only through the divergence at those positions.
- → On-policy *data*, supervised *objective*.

*Which $\beta$?*
- Forward KL ($\beta\to0$) → the student hedges; safe when capacity is close to the teacher's.
- Reverse KL ($\beta\to1$) → the student commits to modes it can actually represent; better under a large capacity gap.
- Reported best settings are task-dependent → treat $\beta$ & $\lambda$ as a 2D sweep, ❌constants.

*Cost vs sequence-level KD?*
- Student generation **every step** + a teacher forward pass on every generated token.
- → Order-of-magnitude more expensive; buy it only when exposure bias is the observed failure.

*Relation to RFT?*
- Both train on self-generated text.
- RFT: verifier filters, target is the sample itself (one-hot), rejected samples discarded.
- GKD: no filter, target is the teacher's full distribution, every sample used.
- → RFT needs a verifier; GKD needs a teacher.
```

&nbsp;

## Preference Optimization
- **What**: Fitting a policy to pairwise comparisons w/ a supervised loss.
- **Why**: SFT cannot express "this is better than that".
    - Its gradient only pushes probability **up** → no mechanism to push a bad response down.
    - Quality is unratable in absolutes but reliably rankable in pairs → the cheap label is a comparison.
    - The [RLHF](rl.md#rlhf) route buys the same signal at the cost of a second model plus a rollout loop.
- **How**:
    1. Collect $(x,y_w,y_l)$ triples.
    2. Define an **implicit reward** as a function of the policy itself → ❌separate RM.
    3. Plug it into a preference likelihood → a classification loss.
    4. Minimize it on the fixed dataset. ❌Sampling, ❌reward model, ❌rollout loop.

```{attention} Q&A
:class: dropdown
*Where does preference data come from?*
- Human labelers comparing 2 responses → expensive, the RLHF standard.
- LM judge → scalable, inherits the judge's biases.
- Constructed → correct vs incorrect answers, strong-model vs weak-model responses, edited vs original.

*Why do on-policy pairs beat off-policy pairs?*
- The loss pushes $\log\pi_\theta(y_l|x)$ **down**; if $y_l$ already has negligible probability under $\pi_\theta$, there is nothing left to suppress.
- Then the gradient mostly reshapes mass elsewhere — off the data entirely.
- → Sample $y_w,y_l$ from the model being trained (or its close relative), then label.

*Why is offline preference optimization structurally weaker than online RL?*
- The dataset is fixed → the objective constrains the policy **only** on responses in $\mathcal{D}$.
- Off that support it says nothing → the optimum is free to put mass on responses no one ever ranked.
- Online RL re-samples, so whatever the policy drifts toward gets graded next step.

*Why is a KL anchor to $\pi_\text{ref}$ present in almost all of them?*
- The preference signal is a **ranking**, which pins down the policy only up to everything it never ranked.
- $\pi_\text{ref}$ supplies the missing prior: stay where you were unless the data says otherwise.
- Methods that drop it (ORPO, SimPO) replace it w/ another anchor — an SFT term or a length-normalized scale.
```

&nbsp;

### DPO
- **Name**: Direct Preference Optimization {cite:p}`rafailov2023direct`
- **What**: Preference classification w/ the LM as its own reward model.
- **Why**: The RLHF stack is expensive & fragile.
    - Separate RM → an extra model to train, store, and overfit.
    - PPO loop → rollouts, critic, clip & KL tuning → the dominant cost & the dominant source of instability.
- **How**:
    1. The KL-constrained RL objective has a **closed-form** optimum.
    2. Invert it → reward = scaled log-ratio of policy to reference.
    3. Substitute into Bradley-Terry → the intractable partition function cancels.
    4. Maximize the likelihood of the observed preferences → binary CE, directly on $\pi_\theta$.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $\pi_\theta$: Policy being trained.
- Hyperparams:
    - $\pi_\text{ref}$: Frozen reference policy.
    - $\beta$: Implicit reward scale (= the KL coeff of the RL problem it replaces).
- Misc:
    - $\hat{r}_\theta$: Implicit reward.

Implicit reward:

$$
\hat{r}_\theta(x,y)=\beta\log\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}
$$

Objective:

$$
\mathcal{L}_\text{DPO}(\theta)=-\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)}-\beta\log\frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}\right)\right]
$$

Gradient:

$$
\nabla_\theta\mathcal{L}_\text{DPO}=-\beta\,\mathbb{E}\left[\underbrace{\sigma\left(\hat{r}_\theta(x,y_l)-\hat{r}_\theta(x,y_w)\right)}_{\text{how wrong the implicit RM is}}\left(\nabla_\theta\log\pi_\theta(y_w|x)-\nabla_\theta\log\pi_\theta(y_l|x)\right)\right]
$$
```

```{tip} Derivation
:class: dropdown
*Where does the objective come from, and why does the reward model vanish?*

1. The KL-constrained RL objective:

    $$
    \max_\pi\ \mathbb{E}_{x\sim\mathcal{D},\ y\sim\pi(\cdot|x)}\left[r(x,y)\right]-\beta\,\text{KL}\left(\pi(\cdot|x)\ \|\ \pi_\text{ref}(\cdot|x)\right)
    $$

2. Its optimum is available in closed form (Gibbs / exponential tilting of the reference):

    $$
    \pi^*(y|x)=\frac{1}{Z(x)}\pi_\text{ref}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right),\qquad Z(x)=\sum_{y}\pi_\text{ref}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right)
    $$

    - $Z(x)$: Partition function — a sum over **all** sequences, intractable.

3. Solve for the reward:

    $$
    r(x,y)=\beta\log\frac{\pi^*(y|x)}{\pi_\text{ref}(y|x)}+\beta\log Z(x)
    $$

4. Bradley-Terry depends only on reward **differences** at the same $x$:

    $$
    p(y_w\succ y_l|x)=\sigma\left(r(x,y_w)-r(x,y_l)\right)
    $$

    → $\beta\log Z(x)$ appears in both terms & cancels. ✅Tractable.

5. Substitute $\pi^*\to\pi_\theta$ & maximize the likelihood of the observed preferences → $\mathcal{L}_\text{DPO}$.

→ The LM **is** the reward model: any $\pi_\theta$ implicitly defines a reward, & fitting the preferences fits the policy directly.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn.functional as F

def seq_logp(logits, labels, ignore=-100):
    ## sum of token log-probs over the RESPONSE tokens -> log pi(y|x)
    logits, labels = logits[:, :-1], labels[:, 1:]
    keep = labels != ignore
    safe = labels.masked_fill(~keep, 0)
    lp = F.log_softmax(logits.float(), -1).gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    return (lp * keep).sum(-1)                       ## (B,)

def dpo_loss(pol_w, pol_l, ref_w, ref_l, beta=0.1):
    ## all four args are seq_logp outputs; ref_* come from the FROZEN model (no grad)
    logits = beta * ((pol_w - ref_w) - (pol_l - ref_l))  ## implicit reward margin
    loss = -F.logsigmoid(logits).mean()
    ## diagnostics that matter: accuracy should rise, both rewards typically FALL
    acc = (logits > 0).float().mean()
    return loss, acc, beta * (pol_w - ref_w).mean(), beta * (pol_l - ref_l).mean()

## Example
pw, pl = torch.tensor([-12.0, -20.0]), torch.tensor([-15.0, -18.0])
rw, rl = torch.tensor([-13.0, -19.0]), torch.tensor([-14.0, -19.0])
print([round(t.item(), 4) for t in dpo_loss(pw, pl, rw, rl)])
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- ⬇️Compute ← ❌RM training, ❌sampling loop; 2 models resident instead of 4.
- Stable ← a plain supervised classification loss w/ a bounded gradient.
- Reuses the SFT trainer; the only new machinery is a frozen forward pass.

*Cons?*
- Off-policy → bounded by the coverage of $\mathcal{D}$.
- Both $\log\pi_\theta(y_w|x)$ **and** $\log\pi_\theta(y_l|x)$ usually fall during training.
- Length bias → longer responses tend to win.
- Overfits preference pairs quickly; 1 epoch is standard.

*Why does the chosen response's log-prob go down?*
- The loss constrains only the **difference** $\hat{r}_\theta(x,y_w)-\hat{r}_\theta(x,y_l)$, never either level.
- $y_w$ & $y_l$ share most of their tokens (prefix, style, formatting) → suppressing $y_l$ drags those shared tokens down too.
- The displaced probability mass goes to sequences that appear in **neither** → completely unconstrained by the objective.
- → Track reward accuracy & the two reward levels separately; falling accuracy w/ a growing margin is the failure signature.

*Why is $\pi_\text{ref}$ necessary?*
- It is the KL anchor inherited from step 1 of the derivation: it makes the reward a *relative* quantity & bounds the drift.
- W/o it, the margin can be maximized by degenerate solutions ($\pi_\theta(y_l|x)\to0$ on everything that resembles $y_l$).
- It also cancels prompt difficulty: a prompt where all responses are improbable does not dominate the loss.

*What must $\pi_\text{ref}$ be?*
- The derivation assumes the preference data was generated by a policy close to $\pi_\text{ref}$.
- → Standard recipe: SFT on the chosen responses first, then use **that** checkpoint as both $\pi_\text{ref}$ & the init.
- Using an arbitrary checkpoint silently violates the assumption.

*What does $\beta$ control?*
- The implicit-reward scale = the KL strength. ⬇️$\beta$ → more drift from $\pi_\text{ref}$; ⬆️$\beta$ → stay put.
- $0.1$ is the common default; Tülu 3 instead used **length-normalized** DPO w/ $\beta=5$ — the log-ratio is divided by $|y|$, so the scale is not comparable. {cite:p}`lambert2024tulu`

*Is DPO really RL-free?*
- As an algorithm, ✅: ❌reward model, ❌sampling, ❌policy gradient — just BCE on a fixed dataset.
- As a **problem**, ❌: every line of the derivation is the KL-constrained RL objective. DPO removes the loop, not the framing.
```

&nbsp;

#### IPO
- **Name**: Identity Preference Optimization {cite:p}`azar2023general`
- **What**: Squared-loss preference matching w/ a finite target margin, ❌Bradley-Terry.
- **Why**: DPO overfits whenever the preferences are (nearly) deterministic.
    - BT maps a preference **probability** to a reward difference; probability $\to1$ maps to difference $\to\infty$.
    - Real datasets have 1 annotation per pair → the empirical probability **is** 0 or 1.
    - → The optimum drives $\pi_\theta(y_l|x)\to0$ & the KL term stops binding, no matter how large $\beta$ is.
- **How**: Replace $-\log\sigma(\cdot)$ — which always rewards more margin — w/ a squared loss that pins the margin at a finite target.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\tau$: Regularization strength (the role $\beta$ plays in DPO).
- Misc:
    - $h_\pi$: Log-ratio margin.

Margin:

$$
h_\pi(y_w,y_l,x)=\log\frac{\pi_\theta(y_w|x)\,\pi_\text{ref}(y_l|x)}{\pi_\theta(y_l|x)\,\pi_\text{ref}(y_w|x)}
$$

Objective:

$$
\mathcal{L}_\text{IPO}(\theta)=\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\left[\left(h_\pi(y_w,y_l,x)-\frac{\tau^{-1}}{2}\right)^2\right]
$$
```

```{attention} Q&A
:class: dropdown
*Why does a squared loss fix the overfitting?*
- $-\log\sigma(z)$ is strictly decreasing in $z$ → the optimizer is **always** paid to grow the margin.
- $(z-c)^2$ penalizes overshoot → the margin is pulled **to** $c=\frac{\tau^{-1}}{2}$ & held there.
- → $\tau$ actually controls the distance from $\pi_\text{ref}$, which is what it was supposed to do all along.

*What exactly is the $\Psi$PO framing?*
- A general objective: maximize $\mathbb{E}[\Psi(p(y\succ y'))]$ minus a KL term.
- DPO = $\Psi=\log\frac{p}{1-p}$ (unbounded, BT-implied); IPO = $\Psi=\text{identity}$ (bounded, no BT assumption).
- → The pathology is the unboundedness of $\Psi$, ❌anything specific to DPO's algebra.

*When does the difference actually show up?*
- Clean, near-deterministic preferences & many epochs → DPO degrades, IPO holds.
- Noisy preferences (multiple annotators, real disagreement) → BT's assumption is closer to true → the gap narrows.

*Cost vs DPO?*
- Identical: same 2 forward passes, same data. Only the scalar loss changes.
```

&nbsp;

#### KTO
- **Name**: Kahneman-Tversky Optimization {cite:p}`ethayarajh2024kto`
- **What**: Preference optimization from **unpaired** binary labels.
- **Why**: Pairing is the expensive part of preference data.
    - Production feedback arrives as thumbs-up / thumbs-down on single responses, ❌as comparisons.
    - Forcing pairs discards unmatched examples & invents comparisons nobody made.
- **How**: Score each response against a reference point w/ a prospect-theory value function.
    1. Implicit reward = the DPO log-ratio.
    2. Reference point $z_0$ = the KL between policy & reference, estimated from **mismatched** pairs in the microbatch.
    3. Value = a saturating function of (reward $-$ reference point), flipped in sign for undesirable responses.
    4. Weight the desirable & undesirable classes to handle imbalance.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\lambda_D$: Weight on desirable examples.
    - $\lambda_U$: Weight on undesirable examples.
    - $\lambda_y$: $\lambda_D$ or $\lambda_U$, whichever applies to $y$.
- Hyperparams:
    - $\beta$: Risk-aversion / reward scale.
- Misc:
    - $z_0$: Reference point.
    - $v$: Value function.

Implicit reward & reference point:

$$
\hat{r}_\theta(x,y)=\log\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)},\qquad z_0=\text{KL}\left(\pi_\theta(y'|x)\ \|\ \pi_\text{ref}(y'|x)\right)
$$

Value:

$$
v(x,y)=\begin{cases}\lambda_D\,\sigma\left(\beta\left(\hat{r}_\theta(x,y)-z_0\right)\right) & y\text{ desirable}\\ \lambda_U\,\sigma\left(\beta\left(z_0-\hat{r}_\theta(x,y)\right)\right) & y\text{ undesirable}\end{cases}
$$

Objective:

$$
\mathcal{L}_\text{KTO}(\theta)=\mathbb{E}_{(x,y)\sim\mathcal{D}}\left[\lambda_y-v(x,y)\right]
$$

$z_0$ is estimated by shifting outputs within the microbatch to form mismatched pairs, clamped at 0, and is **not** backpropagated through.
```

```{attention} Q&A
:class: dropdown
*Why a reference point at all?*
- Prospect theory: humans evaluate outcomes as gains/losses **relative to a baseline**, ❌in absolute terms.
- Mechanically it blocks the cheapest exploit: raising the reward of a desirable response by uniformly inflating the policy also raises $z_0$ → no progress.
- → The model is forced to learn what specifically makes an output desirable.

*Why not backprop through $z_0$?*
- It exists to control where the loss saturates, ❌to be optimized.
- Differentiating it makes training unstable, & $z_0$ is a biased estimate anyway.

*How are $\lambda_D,\lambda_U$ set?*
- Both default to 1, then adjusted for class imbalance: keep $\frac{\lambda_D n_D}{\lambda_U n_U}\in\left[1,\frac{3}{2}\right]$.
- $n_D,n_U$: #desirable & #undesirable examples.

*What $\beta$?*
- $[0.01,0.10]$ for larger models already SFT'd; $[0.10,1.00]$ for smaller models run through KTO w/o prior SFT.

*What is the practical payoff?*
- Uses 1-sided feedback → an order of magnitude more usable data in a deployed product.
- Tolerates class imbalance explicitly, which pairwise losses cannot express.

*What is lost vs DPO?*
- The pairwise signal is the cleanest form of preference; splitting a pair into 2 independent labels discards the fact that they answered the **same** prompt.
```

&nbsp;

#### ORPO
- **Name**: Odds Ratio Preference Optimization {cite:p}`hong2024orpo`
- **What**: SFT loss + an odds-ratio penalty, one stage, ❌reference model.
- **Why**: SFT has a side effect that a second stage then has to undo.
    - SFT raises the likelihood of the chosen response — and of stylistically similar **rejected** responses along w/ it, ← it has no repulsive term at all.
    - Two stages also mean two datasets, two hyperparameter sets, and a frozen copy of the model in memory.
- **How**:
    1. Length-normalize the sequence log-likelihood → $P_\theta(y|x)$.
    2. Odds $=\frac{P}{1-P}$; the log ratio of chosen-to-rejected odds is the preference score.
    3. Loss = NLL on the chosen response $+\ \lambda\times$ the odds-ratio term. Train from the **base** model in one pass.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\lambda$: Weight on the odds-ratio term.
- Misc:
    - $m$: #tokens in $y$.

Length-normalized likelihood:

$$
\log P_\theta(y|x)=\frac{1}{m}\sum_{t=1}^{m}\log P_\theta(y_t|x,y_{<t})
$$

Odds & odds ratio:

$$
\textbf{odds}_\theta(y|x)=\frac{P_\theta(y|x)}{1-P_\theta(y|x)},\qquad \textbf{OR}_\theta(y_w,y_l)=\frac{\textbf{odds}_\theta(y_w|x)}{\textbf{odds}_\theta(y_l|x)}
$$

Objective:

$$
\mathcal{L}_\text{ORPO}=\mathbb{E}_{(x,y_w,y_l)}\left[\mathcal{L}_\text{SFT}+\lambda\cdot\mathcal{L}_\text{OR}\right],\qquad \mathcal{L}_\text{OR}=-\log\sigma\left(\log\textbf{OR}_\theta(y_w,y_l)\right)
$$
- $\mathcal{L}_\text{SFT}$: Ordinary NLL on $y_w$.
```

```{attention} Q&A
:class: dropdown
*Why odds instead of the plain probability ratio?*
- The odds ratio grows much more slowly than the probability ratio as the two likelihoods separate.
- → A milder repulsion: the rejected response is pushed down without collapsing everything that looks like it.
- The probability-ratio version over-suppresses & degrades generation quality.

*Why keep the SFT term?*
- The odds-ratio term is purely **relative** — it is satisfied by lowering $y_l$ as easily as by raising $y_w$.
- The NLL term pins the chosen response's absolute likelihood → the model still learns to *produce* it.
- → This is also what replaces the missing $\pi_\text{ref}$ anchor.

*What is actually saved?*
- 1 model in memory instead of 2, 1 training run instead of 2, 1 dataset.
- ← The reference model exists only to define a relative reward; ORPO's absolute NLL term serves the same purpose.

*Cons?*
- $\lambda$ is the whole balance between imitation & discrimination, & it is sensitive.
- Length normalization is hard-wired → no way to opt out.
- Less validated at frontier scale than DPO.
```

&nbsp;

#### SimPO
- **Name**: Simple Preference Optimization {cite:p}`meng2024simpo`
- **What**: Length-normalized average log-prob as the reward, w/ a target margin, ❌reference model.
- **Why**: DPO's implicit reward is not the quantity decoding actually ranks by.
    - Beam search & sampling rank sequences roughly by **average** log-likelihood per token.
    - DPO's reward is a **sum** of log-ratios against $\pi_\text{ref}$ → a response can win on the training reward yet lose under the model's own decoding metric.
    - The reference model also costs a second resident copy & a forward pass every step.
- **How**:
    1. Reward = average log-prob per token, scaled by $\beta$. ❌$\pi_\text{ref}$.
    2. Bradley-Terry w/ a **target margin** $\gamma$: the chosen must beat the rejected by at least $\gamma$.
    3. Minimize BCE.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\beta$: Reward scale.
    - $\gamma>0$: Target reward margin.

Reward:

$$
\hat{r}_\text{SimPO}(x,y)=\frac{\beta}{|y|}\log\pi_\theta(y|x)=\frac{\beta}{|y|}\sum_{t=1}^{|y|}\log\pi_\theta(y_t|x,y_{<t})
$$

Objective:

$$
\mathcal{L}_\text{SimPO}(\theta)=-\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\left[\log\sigma\left(\frac{\beta}{|y_w|}\log\pi_\theta(y_w|x)-\frac{\beta}{|y_l|}\log\pi_\theta(y_l|x)-\gamma\right)\right]
$$
```

```{attention} Q&A
:class: dropdown
*Why is length normalization load-bearing?*
- W/o it, the learned reward correlates strongly & positively w/ response length → **length exploitation** (longer, lower-quality output).
- W/ it, the reward difference rises for all pairs regardless of their lengths.
- ⚠️ DPO gets a partial version of this for free: the $\pi_\text{ref}$ ratio cancels much of the length dependence.

*What does $\gamma$ do?*
- Demands a strictly positive gap instead of merely "greater than".
- ⬆️$\gamma$ → reward accuracy rises monotonically, but AlpacaEval 2 win rate rises **then falls**.
- ← Large $\gamma$ flattens the reward distribution & lowers the log-likelihood of winning sequences → degeneration.
- → Tune $\gamma$ against generation quality, ❌against reward accuracy.

*What is the cost of dropping $\pi_\text{ref}$?*
- ❌KL anchor → nothing bounds the drift from the SFT model.
- The average-log-prob reward is bounded above by 0, which limits the most degenerate solutions, but it is not a trust region.
- → An SFT regularization term is a common add-back.

*Why does aligning the reward w/ the decoding metric matter?*
- Training optimizes the quantity you write down; serving selects by a different one.
- Any gap between them is a systematic train-inference mismatch — the same failure class as a mismatched chat template.
```

&nbsp;

````{dropdown} Table: Preference Objectives at a Glance
| Method | Implicit reward | Loss shape | $\pi_\text{ref}$ | Data | Length norm | Delta |
|:--|:--|:--|:--|:--|:--|:--|
| DPO | $\beta\log\frac{\pi_\theta}{\pi_\text{ref}}$ | $-\log\sigma(\Delta)$ | ✅ | Pairs | ❌ (partial, via the ratio) | ❌RM, ❌rollouts |
| IPO | Same | $(\Delta-\frac{\tau^{-1}}{2})^2$ | ✅ | Pairs | ❌ | Bounded margin → ❌overfitting |
| KTO | $\log\frac{\pi_\theta}{\pi_\text{ref}}$ | $\lambda_y-\sigma(\beta(\hat{r}-z_0))$ | ✅ | **Unpaired** labels | ❌ | ❌Pairing requirement |
| ORPO | $\log\textbf{odds}_\theta$ | $\mathcal{L}_\text{SFT}+\lambda\mathcal{L}_\text{OR}$ | ❌ | Pairs | ✅ | 1 stage, 1 model |
| SimPO | $\frac{\beta}{\|y\|}\log\pi_\theta$ | $-\log\sigma(\Delta-\gamma)$ | ❌ | Pairs | ✅ | Reward = decoding metric |

$\Delta$: Implicit reward margin between chosen & rejected.
````

&nbsp;

## PEFT
- **Name**: Parameter-Efficient Fine-Tuning
- **What**: Training a small set of params while the pretrained weights stay frozen.
- **Why**: Full fine-tuning does not scale to many tasks or small budgets.
    - Optimizer state dominates: Adam keeps 2 moments per trainable param → ~12–16 bytes/param **on top of** the weights & gradients.
    - 1 full checkpoint per task → storage = #tasks × model size.
    - Small dataset + all params trainable → fast overfitting & [catastrophic forgetting](../../dl/issues.md#catastrophic-forgetting).
- **How**:
    1. Freeze $W_0$ everywhere.
    2. Inject a small trainable module, or select a small subset of existing params.
    3. Train only those.
    4. Ship deltas (MB) instead of models (GB).

```{attention} Q&A
:class: dropdown
*Why does it work at all?*
- Hypothesis: adaptation to a downstream task has a low **intrinsic dimension** — the required update lives in a tiny subspace of weight space.
- Support is empirical (matching full FT at 0.01–1% trainable params), ❌proved.
- The pretrained model already contains the capability; the update only has to *select* it.

*Where does the memory actually go?*
- Weights: frozen → can even be quantized.
- Gradients + optimizer state: $\propto$ **trainable** params → this is the entire win.
- Activations: essentially unchanged ← you still backprop through the whole frozen network.
- → PEFT does **not** remove the need for gradient checkpointing on long sequences.

*When is full FT still the right call?*
- Large distribution shift (new language, new modality) → CPT-scale change, ❌a low-rank nudge.
- Abundant data & budget, single deployment target.
- → PEFT's gap over full FT widens exactly as the required change grows.

*Can you stack PEFT on top of preference optimization?*
- ✅ Orthogonal: the adapter is a parameterization, the objective is a loss. LoRA + DPO is routine.
- ⚠️ $\pi_\text{ref}$ becomes free — disable the adapters & the same model **is** the reference.
```

&nbsp;

### Adapter
- **What**: Small bottleneck MLPs inserted between transformer sublayers. {cite:p}`houlsby2019parameter`
- **Why**: 1 fine-tuned copy per task is unaffordable to store & serve.
- **How**:
    1. After a sublayer, insert: down-project → nonlinearity → up-project.
    2. Wrap it in a residual connection.
    3. Initialize near-identity so the module starts as a no-op.
    4. Train adapters + layer norms only.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{h}\in\mathbb{R}^{d}$: Sublayer output.
- Params:
    - $D\in\mathbb{R}^{b\times d}$: Down-projection.
    - $U\in\mathbb{R}^{d\times b}$: Up-projection.
- Hyperparams:
    - $b\ll d$: Bottleneck width.
- Misc:
    - $\phi$: Nonlinearity.

Forward:

$$
\mathbf{h}\leftarrow\mathbf{h}+U\,\phi\left(D\mathbf{h}\right)
$$
```

```{attention} Q&A
:class: dropdown
*Why near-identity init?*
- A randomly initialized inserted module corrupts the pretrained function at step 0 → the model has to recover before it can learn.
- Near-zero $U$ makes the branch a no-op initially.

*Why did adapters lose to LoRA?*
- They are **sequential**: the branch must run before the next layer → its cost cannot be folded into $W_0$.
- → Permanent extra depth & inference latency at every layer, every token.
- Worse under model parallelism ← an extra synchronization point per layer.

*What survives of the idea?*
- The bottleneck-plus-residual pattern is everywhere (LoRA is its linear, parallel, mergeable cousin).
- Adapters remain attractive when you want a **nonlinear** task-specific transform.
```

&nbsp;

### LoRA
- **Name**: Low-Rank Adaptation {cite:p}`hu2021lora`
- **What**: Trainable rank-$r$ update added **in parallel** to a frozen weight matrix.
- **Why**: Adapters buy parameter efficiency by paying inference latency.
    - A sequential module cannot be folded into $W_0$ → the cost is permanent.
    - A parallel **linear** branch can be added into the weight matrix after training → free at inference.
- **How**:
    1. Factor the update: $\Delta W=BA$ w/ inner dimension $r\ll\min(d_\text{in},d_\text{out})$.
    2. Init $A$ random, $B=0$ → $\Delta W=0$ at step 0, so the model starts unchanged.
    3. Train $A,B$ only, scaling the branch by $\frac{\alpha}{r}$.
    4. At deploy, merge $W\leftarrow W_0+\frac{\alpha}{r}BA$.

```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{x}\in\mathbb{R}^{d_\text{in}}$: Input vector.
    - $\mathbf{h}\in\mathbb{R}^{d_\text{out}}$: Output vector.
- Params:
    - $A\in\mathbb{R}^{r\times d_\text{in}}$: Down-projection, $A\sim\mathcal{N}(0,\sigma^2)$ at init.
    - $B\in\mathbb{R}^{d_\text{out}\times r}$: Up-projection, $B=0$ at init.
- Hyperparams:
    - $W_0\in\mathbb{R}^{d_\text{out}\times d_\text{in}}$: Frozen pretrained weight.
    - $r$: Rank.
    - $\alpha$: Scaling numerator.

Forward:

$$
\mathbf{h}=W_0\mathbf{x}+\frac{\alpha}{r}BA\mathbf{x}
$$

Backward (w/ $\mathbf{g}=\frac{\partial\mathcal{L}}{\partial\mathbf{h}}$):

$$
\frac{\partial\mathcal{L}}{\partial B}=\frac{\alpha}{r}\,\mathbf{g}\left(A\mathbf{x}\right)^T,\qquad \frac{\partial\mathcal{L}}{\partial A}=\frac{\alpha}{r}\,B^T\mathbf{g}\,\mathbf{x}^T,\qquad \frac{\partial\mathcal{L}}{\partial W_0}=\varnothing
$$

Trainable params: $r(d_\text{in}+d_\text{out})$ vs $d_\text{in}d_\text{out}$.
```

````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False               ## W0 is frozen, forever
        d_out, d_in = base.weight.shape
        self.A = nn.Parameter(torch.randn(r, d_in) * 0.01)   ## random -> nonzero grad path
        self.B = nn.Parameter(torch.zeros(d_out, r))         ## zero -> delta W = 0 at step 0
        self.scale = alpha / r                    ## decouples update size from r

    def forward(self, x):
        ## parallel branch: the base output is never modified in place -> mergeable later
        return self.base(x) + self.scale * (x @ self.A.T) @ self.B.T

    @torch.no_grad()
    def merge(self):
        ## fold into W0 -> zero added latency at inference
        self.base.weight += self.scale * (self.B @ self.A)
        return self.base

## Example
layer = LoRALinear(nn.Linear(4, 6), r=2, alpha=4)
x = torch.randn(2, 4)
print(torch.allclose(layer(x), layer.base(x)))    ## True: identical before training
print(sum(p.numel() for p in layer.parameters() if p.requires_grad))  ## 20 vs 24 base weights
```
````

```{attention} Q&A
:class: dropdown
*Pros?*
- On GPT-3 175B vs Adam full FT: 10,000× fewer trainable params & 3× less GPU memory.
- ❌Added inference latency after merging — unlike adapters.
- Adapters are MBs → many task-specific ones can be swapped, or served concurrently against 1 base.

*Cons?*
- Rank caps how much of the update is expressible → lags full FT when the needed change is large.
- Sensitive to $r$, $\alpha$, target modules, & LR simultaneously.
- ⚠️ Saves optimizer memory, ❌activation memory.

*Why $B=0$ & $A$ random, instead of both zero or both random?*
- Both random → $\Delta W\neq0$ at init → the pretrained function is corrupted before training starts.
- Both zero → $\frac{\partial\mathcal{L}}{\partial A}\propto B^T=0$ **and** $\frac{\partial\mathcal{L}}{\partial B}\propto A^T=0$ → dead branch, forever.
- $B=0$, $A\neq0$ → output unchanged, yet $\frac{\partial\mathcal{L}}{\partial B}\neq0$ → $B$ moves first, then $A$ receives gradient.

*What is $\alpha$ for?*
- $\frac{\alpha}{r}$ makes the branch's effective magnitude roughly independent of $r$.
- → Change $r$ w/o re-tuning the LR. Common settings: $\alpha=r$ or $\alpha=2r$.

*Which modules should it target?*
- The original work adapted attention projections only (best results from $W_q,W_v$ at a fixed budget).
- Current practice applies it to **all** linear layers incl. the MLP, which generally helps at equal total params.

*What LR?*
- ~10× the full-FT LR (order $10^{-4}$) ← the branch starts at 0 & has few params to move.
- Reusing the full-FT LR is the most common reason "LoRA didn't learn anything".

*Does the rank need to be large?*
- Style, format, tone, persona → $r=8$–$16$ is usually plenty.
- New knowledge or a real distribution shift → raise $r$, or accept that full FT / CPT is the right tool.

*Why can it be merged when adapters cannot?*
- The branch is **linear** & **parallel** → $W_0\mathbf{x}+\Delta W\mathbf{x}=(W_0+\Delta W)\mathbf{x}$.
- Adapters are sequential & nonlinear → no such algebraic collapse exists.
```

&nbsp;

#### QLoRA
- **What**: LoRA over a 4-bit quantized frozen base. {cite:p}`dettmers2023qlora`
- **Why**: LoRA removes optimizer state, ❌the weights themselves.
    - A 65B model at 16-bit is ~130GB of frozen weights before a single activation is stored.
    - → The base weights, not the trainable ones, become the binding constraint.
- **How**: 3 mechanisms stacked on ordinary LoRA.
    1. **NF4**: A 4-bit NormalFloat data type, information-theoretically optimal for normally distributed weights.
    2. **Double quantization**: Quantize the quantization constants as well.
    3. **Paged optimizers**: Unified memory paging to absorb gradient-checkpointing memory spikes.
    → 65B fine-tuned on a single 48GB GPU while matching 16-bit fine-tuning quality.

```{attention} Q&A
:class: dropdown
*Why doesn't 4-bit destroy quality?*
- The base is **frozen** & used only in the forward pass → quantization error is a fixed perturbation, ❌accumulating noise.
- Weights are dequantized to 16-bit per block for the actual matmul.
- The 16-bit adapters are trained **through** the quantized base → they absorb its error.

*What does it cost?*
- Dequantization on every forward → slower per step than plain LoRA.
- → Trade throughput for the ability to fit the model at all.

*Why NF4 rather than int4?*
- Pretrained weights are approximately zero-centered normal.
- NF4 places its 16 levels at the quantiles of a normal distribution → equal expected mass per level.
- Int4's uniform levels waste resolution in the tails where almost no weights live.

*Can you merge the adapter?*
- ❌ Cleanly into the 4-bit base — merging then re-quantizing discards the adapter's precision.
- → Serve the adapter separately, or merge into the original 16-bit weights.
```

&nbsp;

#### DoRA
- **Name**: Weight-Decomposed Low-Rank Adaptation {cite:p}`liu2024dora`
- **What**: Split each weight into magnitude & direction; apply LoRA to the direction only.
- **Why**: LoRA's update pattern is structurally unlike full FT's.
    - Decompose both into magnitude & direction changes: correlation is $+0.83$ for LoRA but $-0.62$ for full FT.
    - → LoRA moves magnitude & direction nearly proportionally; full FT trades one against the other.
    - → LoRA cannot express "large directional change, small magnitude change", which full FT does routinely.
- **How**:
    1. Decompose $W_0$ into a magnitude vector & a unit-norm direction matrix.
    2. Train the magnitude vector directly; adapt the direction w/ LoRA.
    3. Re-normalize the adapted direction each step, then rescale by the magnitude.

```{note} Math
:class: dropdown
Notations:
- Params:
    - $m\in\mathbb{R}^{1\times d_\text{in}}$: Trainable magnitude vector, init $\|W_0\|_c$.
    - $A,B$: LoRA factors, as in LoRA.
- Hyperparams:
    - $W_0$: Frozen pretrained weight.
- Misc:
    - $\|\cdot\|_c$: Column-wise $L_2$ norm (1 scalar per column).

Decomposition:

$$
W=m\frac{V}{\|V\|_c},\qquad m=\|W\|_c
$$
- $V$: Directional component.

Forward weight:

$$
W'=m\frac{W_0+BA}{\|W_0+BA\|_c}
$$

$m$ & $BA$ are trainable; $W_0$ is frozen. The normalization decouples the two updates, which is the entire point.
```

```{attention} Q&A
:class: dropdown
*Why does decoupling help?*
- Under the decomposition, DoRA's magnitude-direction correlation is $-0.31$ vs full FT's $-0.62$ & LoRA's $+0.83$.
- → It recovers full FT's qualitative learning pattern at LoRA's parameter budget.

*What does it cost?*
- The column-norm & renormalization every step → extra training compute & memory over LoRA.
- ✅Still mergeable → ❌inference overhead.

*When is the gain largest?*
- Low rank. DoRA degrades much more gracefully than LoRA as $r$ shrinks.
- → Useful precisely where LoRA's expressivity limit binds.
```

&nbsp;

### Prefix Tuning
- **What**: Trainable "virtual token" vectors prepended to the keys & values at every layer. {cite:p}`li2021prefix`
- **Why**: Prompting is limited to strings the tokenizer can produce.
    - Discrete prompt search ranges over a finite vocabulary; the optimal conditioning vector need not be any word.
    - Full FT stores a whole model per task.
- **How**:
    1. Prepend $p$ trainable key/value vectors per layer.
    2. Every position attends over [prefix; sequence] → the prefix conditions the whole forward pass.
    3. Reparameterize the prefix through an MLP during training, discard the MLP afterward.
    4. Train ~0.1% of params.

```{attention} Q&A
:class: dropdown
*Why the MLP reparameterization?*
- Optimizing the prefix vectors directly is unstable & highly LR-sensitive.
- Generating them from a smaller matrix through an MLP smooths the optimization; only the generated vectors are kept.

*Why does it beat full FT in low-data settings?*
- Far fewer trainable params → a much stronger implicit prior toward the pretrained function.
- Reported to extrapolate better to topics unseen during training.

*Cons?*
- The prefix permanently occupies context & KV cache → ⬇️usable context window.
- ❌Mergeable → the overhead is per-token, forever.
- More sensitive than LoRA & harder to tune.
```

&nbsp;

#### Prompt Tuning
- **What**: Trainable soft prompt at the **input embedding layer** only. {cite:p}`lester2021power`
- **Why**: Prefix tuning injects at every layer → more params & more plumbing than the effect requires.
- **How**: Prepend $p$ trainable embedding vectors to the input embeddings; freeze everything else, incl. all attention layers.

```{attention} Q&A
:class: dropdown
*When does it actually work?*
- Only at scale: the gap to full FT closes as the model passes ~$10^{10}$ params, and is large below that.
- → "The power of scale": the bigger the frozen model, the less you need to touch it.

*Pros?*
- The smallest footprint of any PEFT method — a handful of vectors per task.
- Enables **prompt ensembling**: many prompts, 1 frozen model, batched together.
- ⬆️Robustness under domain transfer vs full FT.

*Cons?*
- Slow convergence & init-sensitive (initializing from real token embeddings helps).
- Eats context, cannot be merged.
- Weakest expressivity of the family ← it can only re-condition the input, never change the computation.
```

&nbsp;

````{dropdown} Table: PEFT at a Glance
| Method | Trains | Mergeable | Inference overhead | Note |
|:--|:--|:--|:--|:--|
| Full FT | Everything | — | ❌ | The reference point; max capacity, max cost |
| Adapter | Inserted bottleneck MLPs | ❌ | ✅ Latency at every layer | Nonlinear, sequential |
| LoRA | $A,B$ in parallel w/ $W_0$ | ✅ | ❌ | The default |
| QLoRA | LoRA over a 4-bit base | ⚠️ Not into the 4-bit base | ❌ (⬆️ if unmerged) | Fits the model, ⬇️throughput |
| DoRA | Magnitude vector + LoRA direction | ✅ | ❌ | ⬆️Low-rank quality, ⬆️train cost |
| Prefix Tuning | Per-layer KV prefixes | ❌ | ✅ Context + KV cache | ~0.1% of params |
| Prompt Tuning | Input-embedding prompt | ❌ | ✅ Context | Needs a very large frozen model |
| BitFit | Bias terms only {cite:p}`benzaken2021bitfit` | — (already in $W_0$) | ❌ | Smallest possible change to existing params |
| IA³ | Learned rescaling vectors for K, V, FFN {cite:p}`liu2022fewshot` | ✅ | ❌ | Even fewer params than LoRA |
````

&nbsp;

## Practice
### Data
- **What**: The dataset, not the loss, is the method.
- **Why**: The objective is fixed & has no free parameters describing *behavior*.
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
