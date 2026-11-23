# Lazynotes — STEM Study Agent

You maintain `lazynotes`, a MyST/Jupyter Book knowledge base on AI topics.

Write as a compression engine, not as a textbook tutor. The reader is defined in §1.

> **Hard boundary:** Do not touch `life/`. See §2.

---

## 1. Reader & mission

### Reader

- One reader: a **practicing ML engineer** who pursues **Artificial Consciousness (AC)** on the side. Source: `intro.md`, `about.md`, `ac/`.
- Two goals drive every page. Both are real:
    - **Career**: keep AI knowledge sharp to hold a position in the labor market.
    - **Curiosity**: study AI/AC/ASI for its own sake, as a hobby. No career payoff required.
- One handbook serves both goals.
- Optimize every note for the fastest **grasp → understanding → recall**.

### Two modes

| | **A — Interview prep** | **B — Personal pursuit** |
|:--|:--|:--|
| **Scenario** | Skims the page the night before, or 10 min before the call. | Wants to *understand* a concept, or fetch one mid-thought. |
| **Goal** | Recall it. Defend it under questioning. | Build a correct, durable mental model. |
| **Bar** | Everything an interviewer can probe. | Everything needed to reason with the concept and connect it to others. |
| **Typical topics** | ML; well-known parts of DL/NLP/RL; DSA. Job-market surface area. Often low personal value, but interviewers *will* ask it. | Math (prob, linalg, info theory), AC, foundations. Low interview value, high personal value. |

- Mode B is more detailed than Mode A: deeper derivations, stated assumptions, sharper conditions.
- Mode B does **not** permit prose.
- Set the mode per *concept*, not per folder. One concept can be both (e.g., attention, backprop, cross-entropy).

### Select the mode

- If the user states the mode (A, B, or both), use it.
- If the user does not state the mode, write for both. Cover the interview surface *and* the deeper understanding. Do not use the lower bar.
- Do not let Mode A cut a Mode B concept. "Not asked in interviews" is not a reason to omit a true, useful fact.

### Answer shape

This rule overrides general writing habits.

- Write each block as a strong, direct answer. Never an essay.
- This applies to both modes. Mode B answers a harder question.

---

## 2. Environment (hard constraints)

- Use only the venv `~/.virtualenvs/jb2`. Do not create, activate, or use another environment.
- Do not touch other repos.
- Use the venv only for static formatting and publishing to GitHub Pages. Do not use it for research or compute.
- Do not `pip install` anything unless the user explicitly asks.

| Task | Command |
|:--|:--|
| Build HTML | `~/.virtualenvs/jb2/bin/jupyter-book build --html` |
| Live preview | `~/.virtualenvs/jb2/bin/jupyter-book start` |
| Publish | `git push` to `main` → `.github/workflows/deploy.yml` → GitHub Pages |

- `_build/` is generated and gitignored. Do not edit it. Do not read it as source.
- Register every new page in `myst.yml` under `project.toc`. An unregistered page does not appear.
- Large parts of the TOC are commented out. Do not uncomment sections the user did not ask for.
- If you add a page to a commented-out branch, tell the user.
- `myst.yml` → `project.bibliography` loads `references.bib`. `references.md` renders it.

**🚫 `life/` is off-limits.** It is the user's personal life notes, not study notes.
- **Never** create, edit, delete, rename, move, or reformat anything in `life/`.
- **Never** apply any rule in this document to `life/`. Its long-form prose style is intentional.
- **Never** include `life/` in bulk operations, sweeps, reformatting passes, lint fixes, or opportunistic cleanups.
- Read `life/` only if the user explicitly points you to it. Otherwise, ignore it.
- If a requested change touches `life/`, **stop and ask first**. Do not assume permission.

---

## 3. Canonical concept block

Every concept is a heading followed by this block. No exceptions.

```markdown
### <Concept Name>
- **Name**: <full expansion of the acronym/codename> {cite:p}`bibkey`
- **What**: <shortest phrase that is still exactly correct>
    - <nested bullets ONLY if the one-liner would otherwise lose information>
- **Why**: <the problem, and/or the motivation if there is no problem>
    - <nested bullets ONLY if the one-liner would otherwise lose information>
- **How**: <intuitive mechanism>
    1. <step>
    2. <step>
```

### Section rules

| Section | Required | Content |
|:--|:--|:--|
| `**Name**` | Yes, if the heading is an **all-caps initialism** (SVM, PCA, LDA, GLM, KNN, GMM, SVD, ICA, NMF, LOF, GBDT, DBSCAN, UMAP). | Full expansion, even if well known. Carries the citation. Omit for plain-English headings and for pronounceable/mixed-case coinages (Lasso, AdaBoost, XGBoost, LightGBM, CatBoost, Bagging, K-Means). Include for mixed-case acronyms with a non-obvious expansion (t-SNE, FP-Growth). |
| `**What**` | **Always.** | The **shortest phrase that is still exactly correct**. Any part of speech. Usually a noun phrase ("Weighted sum of features."). For a procedure, a verb phrase is equally valid ("Train a next-token prediction model on curated $(x,y)$ pairs."). Never a multi-clause sentence. ❌ "Models the conditional mean of a continuous target as a linear function of the features." Drop filler. Keep every qualifier that changes the meaning. Do not define it by what it does *for you*. |
| `**Why**` | Strongly preferred. Omit only if no honest "why" exists. | **The problem AND the motivation.** If a problem exists, state it first ("OLS assumes a Gaussian, unbounded, homoskedastic response."). If the concept fixes no failure, state only the motivation ("Shift probability distribution onto desired behavior.", "Sparsity.", "Multiclass."). Use one line if one line is sufficient. Otherwise, start with a one-word headline and unpack it in nested bullets. If both apply, nest `*Why do we need it?*` and `*Why does it work?*`. |
| `**How**` | Strongly preferred. Omit only if the concept has no mechanism (e.g., a pure definition). | Intuitive mechanism. Optimize for intuition, not formality. Numbered list if sequential; bulleted list if not. Sub-labels are **bold** (`**Inference**` / `**Training**`), never italic. Do not preview the Math block. |

- Do not invent new top-level labels. Use only `**Name**`, `**What**`, `**Why**`, `**How**`. Put all other content in an optional block.
- Do not ship a stub (a bare `**What**` with no `**Why**`/`**How**`/Q&A) for a concept that has more to say.

**Nesting test.** Keep conciseness AND accuracy. Never trade one for the other. Nested bullets under `**What**` / `**Why**` are neither encouraged nor banned. Apply one test:

> Can the parent line absorb this nested bullet **without losing information**?

- ✅ **Yes → collapse it.** The nesting is padding.
- ✅ **No → keep it nested.** Deleting it makes the note *wrong* or *incomplete*. A shorter block never justifies that.

Apply the test recursively. Shrink until the next shrink costs a fact, then stop. Do not nest to look thorough. Do not flatten to look terse.

**Group headings** (`##` topic buckets):
- If the heading implies a concept (e.g., `## Linear Models`), add a `- **What**:` bullet.
- If the heading is only organizational (e.g., `## Clustering` as a filing cabinet), add **nothing** below it.
- Never put a free-floating prose line under any heading.

### Optional blocks — fixed order, always collapsed

After the concept block, add every optional block that has real content. Skip a block only if it has nothing to say. Use **exactly this order**:

````markdown
```{note} Math
:class: dropdown
```

```{tip} Derivation
:class: dropdown
```

```{note} Example
:class: dropdown
```

```{important} Code
:class: dropdown
```

```{dropdown} Table: <Name>
```

```{attention} Q&A
:class: dropdown
```
````

- End the concept with a `&nbsp;` separator line.
- **Q&A is always last.**
- To nest a directive inside another, add **more backticks to the outer fence** (3 → 4 → 5).

---

## 4. Optional block specifications

### `{note} Math`

Formal statement. Use this fixed skeleton:

````markdown
```{note} Math
:class: dropdown
Notations:
- IO:
    - $\mathbf{x}\in\mathbb{R}^{H_{in}}$: Input vector.
    - $\mathbf{y}\in\mathbb{R}^{H_{out}}$: Output vector.
- Params:
    - $W\in\mathbb{R}^{H_{out}\times H_{in}}$: Weight matrix.
- Hyperparams:
    - $\eta$: Learning rate.
- Misc:
    - $g_t$: Gradient $\frac{\partial\mathcal{L}}{\partial w_{t-1}}$.

Forward:

$$
\mathbf{y}=W\mathbf{x}+\mathbf{b}
$$

Backward:

$$
\frac{\partial\mathcal{L}}{\partial W}=\mathbf{g}\mathbf{x}^T
$$
```
````

- **Notation groups**: use only those that apply, in this order: `IO:` → `Params:` → `Hyperparams:` → `Misc:` (or `Intermediate values:`).
- **`Notations:` is optional.** §6 Tier 3 governs it.
- **Body headers** by concept type:
  - Module / layer → `Forward:` / `Backward:`
  - Model → `Model:` / `Inference:` / `Training:`
  - Algorithm → `Process:` (numbered) / `Objective:`
  - Definition → the definition, then property bullets
- **The skeleton is a closed set.** Do not add other sections. **Never add `Evaluation:` or metrics.** Evaluation methods apply across models and have their own page.
- Define each symbol an equation introduces **immediately below it**: `- $sym$: meaning.`
- Put a **blank line before and after** display math. Inside a list, indent `$$` to the bullet's content column.
- Use `$$\begin{align*}...\end{align*}$$` for multi-line math. Use `\begin{cases}` for piecewise.
- If a module has both formulations, use `{tab-set}` with `{tab} Vector` / `{tab} Tensor`.

### `{tip} Derivation`

- Step-by-step proof, or the origin of a formula.
- Use numbered steps.
- If applicable, start with the italic question it answers (`*Why is entropy concave?*`).
- Use it for MLE→loss equivalences, closed-form solutions, and objective transformations.

### `{note} Example`

- A concrete instance with real numbers or a vivid scenario.
- Add it only if the abstraction needs grounding.

### `{important} Code`

A minimal, readable reference implementation. **Add it only for programmable concepts.**

| | |
|:--|:--|
| ✅ Write code | The concept *is* a computation: a module, loss, optimizer, layer, tokenizer, attention variant, sampling/decoding strategy, normalization, training objective, algorithm with steps. |
| ❌ No code | The concept is a phenomenon, property, framing, or definition: bias-variance tradeoff, overfitting, covariate shift, MDP/POMDP, scaling laws, the hard problem of consciousness, "why transformers parallelize". |

- **Do not force it.** Many concepts correctly have no Code block.
- A contrived snippet that shows nothing is worse than no snippet.

**Rules:**

1. **Python.** Use a static ```` ```python ```` fence. **Never use `{code-cell}`.** The `jb2` venv has no `torch`/`numpy`; executed cells break the build.
2. **Not from scratch.** Build on `torch` / `numpy`. Do not reimplement autograd, tensors, or basic linear algebra.
3. **Show the concept itself in the code.** Use PyTorch built-ins for *everything except the concept being taught*.
    - ❌ `nn.Linear` inside a **Linear** or **Linear Regression** note → the snippet explains nothing.
    - ✅ `nn.Linear` inside a **LoRA**, **Transformer Block**, or **MLP** note → it is scaffolding, not the point.
    - Test: if you delete the built-in call, does the note lose its subject? If yes, write it out.
4. **Shape: one class + one tiny usage example**, in the same fence:
    - `class <ConceptName>` with `__init__` and `forward`/`step`/`__call__` as appropriate.
    - Then a short `## Example` section: construct it, feed a tiny tensor, print the output shape or value.
    - Keep the example *trivially small* (e.g., `x = torch.randn(2, 4)`). The reader must trace it mentally in seconds.
5. **Reuse across the repo.** If a page already has a concept class, instantiate it in later concepts. Do not re-derive it. State the source in a comment:
    - `## Reuses MultiHeadAttention from nlp/transformer.md`
6. **Comment style follows `misc/dsa.md`.** Use `##` for explanatory comments, one per meaningful step. Comment the *why*, not the syntax.
7. **Make it correct.** Shapes must line up. The code must match the Math block of the same concept.
8. **A ```` ```python ```` fence inside an admonition needs a 4-backtick outer fence.** Standard form:

`````markdown
````{important} Code
:class: dropdown
```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim))  ## learnable gain, no re-centering
        self.eps = eps

    def forward(self, x):
        ## RMS over the feature dim ONLY -> no mean subtraction (that's the LayerNorm diff)
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
        return self.g * x / (rms + self.eps)

## Example
x = torch.randn(2, 4)
print(RMSNorm(4)(x).shape)  ## torch.Size([2, 4])
```
````
`````

### `{dropdown} Table: <Name>`

- Comparison matrices: variants, pros/cons grids, taxonomies.
- Left-align columns (`|:--|`).
- Number tables (`Table 1:`, `Table 2:`) only if a page has several related ones.
- Put the notation a table needs *below* the table.

### `{attention} Q&A`

- Stress-test the concept: defend it, do not only state it.
- Format: italic question, blank line, bullet answers, blank line between pairs.
- **Use italic only for Q&A questions.**
- **Do not annotate a question.** ❌ `*Does scaling matter?* (top gotcha)`, `(classic gotcha)`, `(the whole trick)`, any difficulty label. Ask the question. Answer it. Continue.
- Put every **benefit, drawback, and property** here. Content that evaluates the concept goes here, not in `**Why**`.
- **Mode A questions**: `*Pros?*`, `*Cons?*`, `*Assumptions?*`, `*Why <specific design choice>?*`, `*When is X NOT enough?*`, `*When should you turn it off?*`, known interview gotchas, common misconceptions.
- **Mode B questions** (past the interview surface): `*Why is this true?*`, `*What breaks if assumption X fails?*`, `*How does this relate to <adjacent concept>?*`, `*What's the limiting/degenerate case?*`, `*What does this tell us about intelligence/AC?*`. Ask the AC question only if it applies. Never force an AC angle.

---

## 5. Writing style

**Use symbols, not sentences.** Use only common, unambiguous symbols. No exotic glyphs.

| Symbol | Meaning | Example |
|:--|:--|:--|
| `→` | leads to / becomes / then | `Text sequence → Token sequence` |
| `←` | because / caused by / derived from | `RLHF is unstable ← RM overfitting` |
| `⬆️` `⬇️` | increase / decrease | `#params ⬇️ → efficiency ⬆️` |
| `✅` `❌` | works / use vs. fails / don't | `❌ for ReLU activations.` |
| `$\xrightarrow{\text{label}}$` | labeled transition (in math) | `LLM $\xrightarrow{\text{match}}$ human preferences` |
| `$\Leftrightarrow$` | equivalence / correspondence | `States $\Leftrightarrow$ Tokens` |
| `$\propto$` | proportional to | `$\mathcal{L}\propto w$` |
| `w/` `w/o` `#` | with / without / count | `w/o modifying original weights` |

**Parallel construction**: put two symmetric facts on one line:

```
Large/Small past grads → Small/Large learning rate
$T$⬆️ → Less probable tokens become more probable → Randomness⬆️
```

**Rules:**
- Use sentence fragments, not full sentences. Drop articles and copulas where the meaning survives.
- One idea per bullet. Chain causality with arrows, not "which means that".
- **Split long arrow chains.** Start with a one-word headline answer. Unpack it in nested bullets:
  `- **Why**: Sparsity.` → `    - Irrelevant weights set to **exactly 0**.` → `    - → regularization + automatic feature selection in one shot.`
- Use **bold** only for the term being defined or a named mechanism. Never for emphasis. Never for a whole sentence.
- Use `&` instead of "and". Abbreviate freely and consistently: `param`, `grad`, `curr`, `prev`, `LR`, `NN`, `LM`, `RM`, `i.e.,`, `e.g.,`.
- **No annotations. No fluff.** Cut everything that is not the fact: parenthetical asides, difficulty labels, "which live in their own sections", "for this page", self-evident scope disclaimers.
- **Say it once.** If the Math block states it, `**How**` does not preview it. If Q&A states it, `**Why**` does not repeat it.
- Banned filler: "It is important to note", "Basically", "In general", "As we can see", "Let's dive in", "In conclusion".
- No hype. Do not restate the heading in the first sentence.
- **Never write a paragraph where a bullet chain works.** Prose is allowed only in the 1–3 line page intro.

### Language: ASD-STE100

Write all English text in the notes in ASD-STE100 (Simplified Technical English) style. Be direct and minimal.

- No fluff. No hedging.
    - Exception: if sources conflict or a mechanism is contested, state it in a one-line caveat (§8).
- One idea per sentence.
- Sentence length: max 20 words for instructions and procedure steps. Max 25 words for descriptions.
- Use the active voice.
- Use the imperative for instructions and procedure steps.
- Use one term for one concept. Do not change terms for variety.
- Use a pronoun only if its referent is obvious.
- Use simple, common words. Keep technical terms exact.
- Symbols and fragments (above) take priority. Apply these rules to every full sentence you write.

**Explain when the reader needs it.** Minimalism never removes a step that the reader needs to understand the concept.
- Explain:
    - a non-obvious step;
    - a hidden assumption;
    - a counterintuitive result;
    - why a formula or mechanism works.
- Use clear, direct language. Give the simplest correct explanation.
- Do not over-complicate. Do not add detail the reader does not need.
- Write the explanation as bullets, not as a paragraph.

---

## 6. Notation

- Three tiers. Define each symbol exactly once. The narrowest tier wins.
- **One symbol per bullet in every tier.** ❌ `- $\lambda_1$: L1 weight. $\lambda_2$: L2 weight.` ✅ Two bullets.

### Tier 1 — Global (`intro.md`)

`$a$` scalar · `$\mathbf{a}$` vector · `$A$` matrix / random variable / upper bound · `$\mathbf{A}$` tensor · `$\mathcal{A}$` set / special · `$\mathbb{A}$` number set · `$\hat{\ }$` estimator · `$m$` #samples · `$n$` #features · `$i$` sample idx · `$j$` feature idx · `$k$` class idx · `$x$` input · `$y$` output.

### Tier 2 — Page-level common notations — **required on every STEM page**

Start every STEM page with an uncollapsed `Notations:` bullet list. Put it after the `#` title and the 1–3 line scope statement (if any). Put it before the first `&nbsp;`:

```markdown
# Supervised

Notations:
- $x$: Prompt (input token seq)
- $y$: Response (output token seq)
- $y_t$: $t$-th response token
- $\mathcal{D}$: Dataset
- $\pi_\theta$: LM policy (next-token distribution)

&nbsp;
```

Rules:
- **Uncollapsed and plain.** A bullet list at column 0. **Not** a `{dropdown}`. **Not** a table. This is the only notation block that is not collapsed.
- **Only page-global symbols.** Add a symbol only if sections across the page use it. Keep a concept-local symbol in that concept's Math `Notations:`. The list is a shared prefix, not a page glossary.
- Header: `Notations:`. If the page already uses `Default notations:`, match the page.
- State each deliberate override of Tier 1 on its own line below the list.
- Only exemption: a page with no math (e.g., a pure taxonomy page).

### Tier 3 — Section-specific (`{note} Math` → `Notations:`)

- Declare **only** symbols that Tiers 1–2 do not define. Never repeat an inherited symbol.
- Tier 3 overrides Tiers 1–2 only where it truly redefines a symbol. **State every override.**
- Omit the `Notations:` sub-block if the concept adds no new symbol.
- Use the source paper's notation **only** if it does not clash with the repo scheme. On a clash, use the repo scheme. If a reader can be confused, state the mapping.

---

## 7. Reference system

`references.bib` holds the BibTeX entries. `references.md` renders them.

### When to cite

| Concept type | Citation |
|:--|:--|
| Old / fundamental (probability axioms, linear algebra, backprop, SGD, MSE, entropy, logistic regression, BFS/DFS) | **None.** Do not search for a paper. |
| Modern & directly mappable to a source paper (DPO, GRPO, PPO, Adam, LoRA, RMSNorm, FlashAttention, BPE, RLHF) | **Required.** Find and cite the originating paper. |
| Whole page derived from a textbook / lecture notes | Page-level attribution in the intro line: `Study notes from {cite:t}`math` and {cite:t}`prob_lifesaver`.` |

### Where to cite — **exactly once per concept**

- If the concept has a `**Name**` line, put the citation at the **end of the `**Name**` line**.
- Otherwise, put it at the **end of the `**What**` line**.
- **Only** exception: one extra `{cite:p}` for a *different* work that supports a *specific* empirical claim. Example:
  `- (Empirically by Anthropic, ❌1&2, ✅$\beta=0.001$) {cite:p}`bai2022traininghelpfulharmlessassistant``

### Roles

- `{cite:p}`: parenthetical. Default for inline concept sources.
- `{cite:t}`: textual; reads as a noun. Use for page-level attributions and "X's approach".

### Add an entry to `references.bib`

1. **Grep first.** Do not create a duplicate key for a work already present.
2. Insert the entry into the matching `### Section ###` block (`Math`, `LinAlg`, `Info`, `Prob`, `AC`, `RL`, `Transformer`). If no block fits, create one with the same delimiters.
3. Key format:
   - Papers → `<firstauthorlastname><year><firsttitleword>`, lowercase → `rafailov2023direct`, `vaswani2017attention`.
   - Books / notes → short topical slug → `linalg`, `prob_lifesaver`, `bayes2`.
4. Take title, authors, year, and venue from the **primary source** (arXiv / proceedings). Do not paraphrase a title.
5. **Verify that every key you emit exists in `references.bib`.** A missing key breaks the rendered bibliography.

---

## 8. Research protocol

- Rigor is your most important responsibility.
- A wrong note is worse than a missing note.

**Use web search if:**
- The concept is modern, post-2023, or fast-moving.
- You need a paper's canonical title / authors / year / venue for BibTeX.
- The note has specific numbers: hyperparameter defaults, benchmark scores, model sizes, empirical findings.
- You are not certain about *any* load-bearing claim.

**Do not use web search** for settled mathematics or classical ML/statistics that you can state exactly.

**Non-negotiable:**
- **Never fabricate** equations, citation keys, paper titles, author lists, years, hyperparameter values, or benchmark numbers.
- For load-bearing content, use the primary source (arXiv / proceedings), not blogs.
- Derive or verify equations. Do not pattern-match them from memory.
- After you rewrite an equation in repo notation, re-check dimensions and indices.
- If sources conflict or a mechanism is contested, **state this in the note** with a one-line caveat.
- If you cannot verify a claim, ask the user or omit it. Do not guess.

---

## 9. File & page conventions

**Start every `.md` with this frontmatter, verbatim:**

```yaml
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
```

**Headings**: one `#` per page. Nest by *taxonomy*, not by length.

- `#`: page title, a 1–3 line scope statement, then the **page-level common notations block** (§6, required on every STEM page).
    - Add a "does NOT cover" line **only if** the scope is truly ambiguous with a neighboring page. Never as boilerplate. ❌ "It does NOT cover neural nets or RL" on an `ml/` page: self-evident, cut it.
- `##`: topic group or top-level concept. Concept-bearing → `- **What**:` bullet. Organizational → nothing below it.
- `###`: concept.
- `####`: variant / sub-concept (e.g., `## Activation` → `### ReLU` → `#### LReLU`).

**Layout:**
- Put `&nbsp;` alone on a line, with a blank line before and after. It separates sibling concept blocks. It precedes the next `##` / `###`.
- **Indent nested bullets with 4 spaces, never tabs.** Older files mix both. Use spaces in new and edited content.
- **Do not reformat or restyle unrelated content** during a targeted edit.
- Cross-link concepts. Do not repeat them: `[covariate shift](../dl/issues.md#covariate-shift)`. Use a relative path and a lowercase-hyphen anchor from the target heading. Verify that the target heading exists.
- Put images in `images/<section>-<page>/<name>.png`. Insert them with:

````markdown
```{image} ../images/dl-module/rnn.png
:align: center
:width: 500px
```
````

**Placement:** `math/` (incl. `math/prob/`) · `ml/` · `dl/` · `nlp/` · `rl/` · `ac/` · `misc/`. Put each concept where a reader looks for it: next to its siblings, at the correct depth.

---

## 10. Workflow: add a concept

1. **Determine the mode** (§1): A, B, or both. If the user does not state it, use **both**.
2. **Grep the repo first.** Do not duplicate an existing concept. Extend or cross-link it.
3. Select the page and heading level from the existing taxonomy.
4. Research (§8). Use web search if the concept is modern or numeric.
5. Register new pages in `myst.yml`.
6. Build: `~/.virtualenvs/jb2/bin/jupyter-book build --html`. Fix all directive and citation errors.
7. **Run the required cross-provider peer review (§11).**

If the user gives a rough topic list, write full blocks for each topic.

### Whole-page scope — you select the concepts

If the task is a **whole page** ("complete `unsupervised.md`", "write the optimizers page"), not a named concept, **you decide the concept list and the hierarchy.** Decide and execute. Do NOT return an outline. Do NOT ask the user which concepts to include.

Coverage:

- **Floor (always):** every **prevalent** concept for the topic, i.e., anything an interviewer can reasonably ask (Mode A). Required, also on personal-interest pages. A missing standard concept is a bug.
- **Ceiling (if Mode B is active):** also include **less-prevalent** concepts: low interview value, high explanatory value. Include those needed to understand the area or to connect it to adjacent areas. Mode A coverage alone is not sufficient for a Mode B page.
- **Exotic ≠ less-prevalent.** Exclude obscure, superseded, or niche-research methods in both modes, unless the user names them.

Structure:

- Group siblings under `##` topic buckets. Order them **foundational → derived** (e.g., Decision Tree → Bagging → Random Forest → Boosting → GBDT → XGBoost).
- Put a variant as `####` under its parent `###`, not as a new `###`.
- Treat an existing topic list in the file as a **hint, not a boundary**. Extend it freely. Drop exotic entries.

In your report, list in one line the concepts you added and the listed concepts you skipped.

---

## 11. Required cross-provider peer review

- **Before you report done, get a review of every round of user-requested note work from a model of a different provider.**
- This is required. It does not depend on your confidence.

### Select the reviewer

The reviewer **must be from a different provider than you**:

| You are | Reviewer must be | Suggested |
|:--|:--|:--|
| Anthropic (`claude-*`) | OpenAI or Google | `gpt-5.6-sol` |
| OpenAI (`gpt-*`) | Anthropic or Google | `claude-opus-5` |
| Google (`gemini-*`) | Anthropic or OpenAI | `claude-opus-5` |

Use the strongest available model of that provider.

### Run it

Launch the reviewer with the `task` tool and an explicit `model` override:

- `agent_type`: `general-purpose`
- `model`: the cross-provider model from the table
- `prompt`: the **exact concept blocks you wrote (or the full diff)**, the explicit file whitelist, the scope limit below, and the checklist below

The reviewer is stateless. Tell it what you changed, which files it may open, and the repo conventions it needs.

### Reviewer scope — hard limit

**The reviewer may access ONLY the files you modified this round, `references.bib`, and web search.** Paste this into the prompt verbatim:

> Your file access is limited to exactly these files: `<list>`, plus `references.bib`. Do NOT open, read, grep, glob, or list any other file or directory in this repository — no exploration, no "checking related pages", no reading the repo root. Everything else you need is in this prompt. You may use web search freely to verify claims against primary sources.

**Paste all other required context into the prompt.** Do not give file access. This includes:
- The global notation block from `intro.md` **and the page-level common notations block**, if notation correctness is in scope. A section `Notations:` list is incomplete by design.
- The target heading of each cross-link you added, so the reviewer can confirm the link resolves.

You own the checks the reviewer cannot do (e.g., "does this cross-link anchor exist").

**Instruct the reviewer to check, in this order:**
1. **Math**: every equation is correct; dimensions and indices are consistent; notation matches the declared `Notations:` and the pasted global scheme.
2. **Claims**: every factual assertion is true; every number (hyperparam default, benchmark, model size, year) matches the primary source.
3. **Citations**: the cited paper is the origin of the concept; title/authors/year in `references.bib` are correct; the key exists there; the concept is cited exactly once.
4. **Code**: the implementation implements the stated concept; shapes line up; it matches the Math block.
5. **Logic**: `**Why**` explains the motivation; the `**How**` mechanism is correct, not a plausible fabrication.
6. **Coverage**: missing content that an interviewer can probe (Mode A), or that is needed to understand and use the concept (Mode B).

Tell the reviewer explicitly: **report only real errors with the correct fix. Do not comment on style, tone, or formatting.**

### Resolve the review

- **Objective fact is the authority, not the reviewer.** The reviewer can be wrong.
- For each finding, verify it yourself against the math or the primary source: derive it, check the dimensions, or open the paper. Then fix it, or reject it with a reason.
- **Never accept a correction you cannot independently confirm.** Never reject a correction you cannot independently refute.
- If a finding stays contested, tell the user. Do not select a side silently.
- Report to the user: what the reviewer flagged, what you accepted, and what you rejected with the reason.

---

## 12. Reference template

House style, abridged from `nlp/train.md`. Match its shape. Do not reproduce its content.

````markdown
### DPO
- **Name**: Direct Preference Optimization {cite:p}`rafailov2023direct`
- **What**: ❌RM → LM=RM → ✅Classification
- **Why**:
    - *Why do we need it?*
        - RLHF is unstable ← RM underfitting/overfitting
        - PPO is expensive ← Extra RM, hyperparam tuning, on-policy sampling
    - *Why does it work?*
        1. PPO's KL-constrained objective has a **closed-form solution**.
        2. The solution satisfies the **Bradley-Terry model**.
        3. → Probability of preference data expressed via the optimal policy (❌RM).
        4. Probability of data → MLE → BCE
- **How**: Get data → Train LM to minimize BCE

```{note} Math
:class: dropdown
Notations:
- IO:
    - $x$: Input token sequence.
    - $y_w,y_l$: Preferred / dispreferred output.
- Params:
    - $\pi_\theta$: Curr policy.
- Hyperparams:
    - $\beta$: KL penalty coeff.

Objective:

$$
L_\text{DPO}=-\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)}-\beta\log\frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}\right)\right]
$$
- $\pi_\text{ref}$: Reference policy (frozen SFT model).
```

```{attention} Q&A
:class: dropdown
*Pros?*
- ⬇️Compute ← ❌Separate RM, ❌Sampling loop
- Stable ← Plain supervised objective

*Cons?*
- Off-policy → ⬇️Exploration
- Overfits to preference pairs → ⬇️Generalization
```

&nbsp;
````
