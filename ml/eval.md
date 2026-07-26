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
# Evaluation
How a trained model is scored. [Objectives](obj.md) are what it minimizes; metrics are what gets reported, thresholded & selected on.

Task-generic **offline** metrics ONLY.
- ❌ Field-specific: NLP (BLEU/ROUGE/perplexity), CV (IoU/mAP), RL (return/regret).
- ❌ Online metrics & A/B testing → [Misc](misc.md#online-a-b-testing).

Default notations:
- $m$: #samples.
- $K$: #classes.
- $y_i$: True label/value of sample $i$.
- $\hat{y}_i$: Predicted label/value of sample $i$.
- $s_i\in[0,1]$: Predicted score for the positive class.
- $\hat{p}_{ik}$: Predicted probability of class $k$ for sample $i$.
- $\tau$: Decision threshold, $\hat{y}_i=\mathbb{1}[s_i\ge\tau]$.
- $\pi$: Prevalence, i.e., fraction of positives.
- TP, FP, FN, TN: Confusion counts, defined in [Confusion Matrix](#confusion-matrix).

&nbsp;

## Framework
### Metric vs Loss
- **What**: Reported score vs optimized objective.
- **Why**: The quantity a decision actually depends on is piecewise-constant (accuracy) or set-level (F1, AUC) → ❌gradient, ❌minibatch decomposition → unusable for training.
- **How**:
    1. Fix the metric from the decision cost.
    2. Train on a differentiable [surrogate](obj.md#surrogate-loss).
    3. Select model, hyperparams & $\tau$ **on the metric**, on validation data.
    4. Report on test data.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $C_\text{FP}$: Cost of a false positive.
    - $C_\text{FN}$: Cost of a false negative.
- Misc:
    - $s=P(y=1|\mathbf{x})$: Calibrated positive-class probability.

Expected cost of predicting $\hat{y}$ for a single sample:

$$\begin{align*}
\mathbb{E}[\text{cost}|\hat{y}=1]&=(1-s)C_\text{FP}\\
\mathbb{E}[\text{cost}|\hat{y}=0]&=sC_\text{FN}
\end{align*}$$

Cost-optimal threshold:

$$
\tau^*=\frac{C_\text{FP}}{C_\text{FP}+C_\text{FN}}
$$
- General $K$-class cost matrix → [0-1 Loss](obj.md#id-0-1-loss).
```

```{tip} Derivation
:class: dropdown
*Where does $\tau^*$ come from?*
1. Predict $1$ iff it is the cheaper action: $(1-s)C_\text{FP}<sC_\text{FN}$.
2. Expand: $C_\text{FP}-sC_\text{FP}<sC_\text{FN}$.
3. Collect: $C_\text{FP}<s(C_\text{FP}+C_\text{FN})$.
4. → $s>\frac{C_\text{FP}}{C_\text{FP}+C_\text{FN}}=\tau^*$.
5. $C_\text{FP}=C_\text{FN}$ → $\tau^*=0.5$. Equal costs are the ONLY reason $0.5$ is ever the right threshold.
```

```{dropdown} Table: Metric Selection
| Situation | Metric |
|:--|:--|
| Balanced classes, symmetric costs | Accuracy |
| Imbalanced, minority class is the point | PR-AUC, macro-F1 |
| Threshold-free ranking quality | ROC-AUC |
| Probabilities feed a downstream decision | Log Loss, Brier, ECE |
| Known asymmetric costs | Expected cost at $\tau^*$ |
| Multi-class, every class equally important | Macro-F1 |
| Multi-class, every sample equally important | Micro-F1 (= accuracy) |
| Regression, outliers = noise | MAE |
| Regression, large errors disproportionately costly | RMSE |
| Regression, relative error matters | MAPE, RMSLE |
| Regression, "is this better than the mean?" | $R^2$ |
| Clustering, ❌ground truth | Silhouette |
| Clustering, ✅ground truth | ARI, NMI |
```

```{attention} Q&A
:class: dropdown
*Why not just optimize the metric directly?*
- Accuracy/0-1 → $\nabla=0$ a.e.
- F1/AUC/precision → **non-decomposable**: not an average of per-sample terms → a minibatch estimate is biased and high-variance.
- Fix: train on a surrogate, then tune $\tau$ on the metric post-hoc.

*Decomposable vs non-decomposable metrics?*
- Decomposable: accuracy, log loss, Brier, MSE, MAE → per-sample average → safe to compute on batches & average over folds.
- Non-decomposable: precision, recall, F1, AUC, $R^2$ → must be computed on **pooled** predictions over the whole set. Averaging per-fold F1 ≠ F1 of the pooled folds.

*How many metrics should you report?*
- One **optimizing** metric (the one being maximized) + several **satisficing** constraints (latency, memory, worst-group score, fairness gap) that only need to clear a bar.
- Multiple optimizing metrics → no defined ordering over models → no decision.

*Is a gap between two models real?*
- Point estimates on a finite test set are noisy → report a CI.
- Paired bootstrap over test samples for any metric; McNemar's test for comparing two classifiers on the same set.
- Decide on a CI for the **paired difference**: if it contains $0$, the models are indistinguishable on this data. Comparing the gap against each model's own CI width is not a valid test.

*Why does the metric have to be picked before modeling?*
- Every downstream choice (threshold, class weights, early stopping, feature set) is a selection made *on* the metric → changing it afterwards invalidates the selection.
```

&nbsp;

## Classification
### Confusion Matrix
- **What**: Count table of true vs predicted labels.
- **Why**: A single scalar hides *which* errors occurred → FP & FN carry different costs.
- **How**:
    1. Turn scores into labels: threshold at $\tau$ (binary) or $\arg\max_k$ (multi-class).
    2. Tally every (true, predicted) pair.
    3. Every threshold metric is a ratio of these counts.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $C\in\mathbb{N}^{K\times K}$: Confusion matrix, rows = true, cols = predicted.
    - $\text{TP}$: #samples with $y=1,\hat{y}=1$.
    - $\text{FP}$: #samples with $y=0,\hat{y}=1$.
    - $\text{FN}$: #samples with $y=1,\hat{y}=0$.
    - $\text{TN}$: #samples with $y=0,\hat{y}=0$.

Definition:

$$
C_{kl}=\sum_{i=1}^{m}\mathbb{1}[y_i=k]\mathbb{1}[\hat{y}_i=l]
$$

Binary case:

$$
C=\begin{pmatrix}\text{TN} & \text{FP}\\ \text{FN} & \text{TP}\end{pmatrix}
$$

Properties:
- $m=\text{TP}+\text{FP}+\text{FN}+\text{TN}$.
- Row-normalized → recall per class. Column-normalized → precision per class.
- Row sums are fixed by the data; column sums are chosen by the model → only the columns move with $\tau$.
- Multi-class → binarize one-vs-rest per class $k$: $\text{TP}_k=C_{kk}$, $\text{FP}_k=\sum_{l\neq k}C_{lk}$, $\text{FN}_k=\sum_{l\neq k}C_{kl}$.
```

````{important} Code
:class: dropdown
```python
import numpy as np

def confusion_matrix(y, yhat, K):
    ## flatten each (true, pred) pair into a single index, then count
    return np.bincount(y * K + yhat, minlength=K * K).reshape(K, K)

## Example
y    = np.array([0, 0, 1, 1, 1, 0])
yhat = np.array([0, 1, 1, 1, 0, 0])
print(confusion_matrix(y, yhat, 2))
## [[2 1]   <- TN=2, FP=1
##  [1 2]]  <- FN=1, TP=2
```
````

```{dropdown} Table: Rates
| Rate | Formula | Also called |
|:--|:--|:--|
| TPR | $\frac{\text{TP}}{\text{TP}+\text{FN}}$ | Recall, sensitivity, hit rate |
| TNR | $\frac{\text{TN}}{\text{TN}+\text{FP}}$ | Specificity, selectivity |
| FPR | $\frac{\text{FP}}{\text{FP}+\text{TN}}$ | False alarm rate, $1-$TNR |
| FNR | $\frac{\text{FN}}{\text{FN}+\text{TP}}$ | Miss rate, $1-$TPR |
| PPV | $\frac{\text{TP}}{\text{TP}+\text{FP}}$ | Precision |
| NPV | $\frac{\text{TN}}{\text{TN}+\text{FN}}$ | — |

- TPR & TNR condition on the **truth** → independent of $\pi$.
- PPV & NPV condition on the **prediction** → move with $\pi$.
```

```{attention} Q&A
:class: dropdown
*Why look at it before any scalar metric?*
- $K^2$ numbers → every scalar metric is a lossy projection of them.
- Reveals the failure *pattern*: which class is being confused with which, and whether errors are one-directional.

*How to read a multi-class one?*
- Row $k$ = what the model does with true class $k$ → off-diagonal mass = what $k$ leaks into.
- Column $k$ = what a predicted $k$ is really made of.
- Big off-diagonal blocks → suspect overlapping class definitions or label noise before reaching for more capacity.

*Does it depend on the threshold?*
- Every entry does. Metrics computed from it (accuracy, P/R/F1, MCC) are **operating-point** metrics; ROC/PR sweep $\tau$ to remove that dependence.

*Normalize it?*
- By row for recall-style reading, by column for precision-style reading. Raw counts hide a 5-sample class; row-normalizing hides that the class has 5 samples → show both.
```

&nbsp;

### Accuracy
- **What**: Fraction of correct predictions.
- **Why**: Need one number for "how often is it right" when all error types cost the same.
- **How**: Count matches → divide by $m$.

```{note} Math
:class: dropdown
Definition:

$$
\text{Acc}=\frac{1}{m}\sum_{i=1}^{m}\mathbb{1}[\hat{y}_i=y_i]=\frac{\text{TP}+\text{TN}}{m}=\frac{\text{tr}(C)}{m}
$$

Properties:
- $\text{Acc}=1-$ empirical [0-1 loss](obj.md#id-0-1-loss) → decomposable per sample.
- Range $[0,1]$.
- Baseline = majority-class rate $\max_k\pi_k$, NOT $\frac{1}{K}$ and NOT $0.5$.
```

```{attention} Q&A
:class: dropdown
*Pros?*
- Interpretable with no context, any $K$.
- Decomposable → unbiased on minibatches, averages over folds.
- Symmetric across classes → no arbitrary "positive class" choice.

*Cons?*
- **Accuracy paradox**: $\pi=0.01$ → the constant-negative model scores $99\%$ and is useless.
- Prices FP and FN identically.
- Threshold-dependent, and $\tau=0.5$ is only optimal under equal costs.
- One number for $K$ classes → hides total failure on a rare class.

*When is it the right metric?*
- Roughly balanced classes **and** symmetric costs **and** you care about the hard label, not the probability.

*How to fix it under imbalance?*
- Report against the majority baseline, not against $0$.
- Or switch to balanced accuracy / macro-F1 / MCC.
```

&nbsp;

#### Balanced Accuracy
- **What**: Mean of per-class recalls.
- **Why**: Plain accuracy is a $\pi$-weighted average of per-class recalls → the majority class owns the score.
- **How**: Recall per class → unweighted mean.

```{note} Math
:class: dropdown
Definition:

$$
\text{BA}=\frac{1}{K}\sum_{k=1}^{K}\text{Recall}_k\quad\overset{K=2}{=}\quad\frac{\text{TPR}+\text{TNR}}{2}
$$

Chance-adjusted variant:

$$
\text{BA}_\text{adj}=\frac{\text{BA}-\frac{1}{K}}{1-\frac{1}{K}}
$$

Properties:
- Any constant-class predictor → $\text{BA}=\frac{1}{K}$, regardless of $\pi$.
- Identical to macro-averaged recall.
- Invariant to $\pi$ ← each class is normalized by its own support.
```

```{attention} Q&A
:class: dropdown
*Pros?*
- Immune to the accuracy paradox — the degenerate classifier scores $0.5$ instead of $0.99$.
- Same interpretation for any $K$.

*Cons?*
- Ignores precision entirely → a flood of FPs is invisible as long as TNR stays high (which it does when negatives are abundant).
- Implicitly reweights the test set to uniform priors → no longer reflects the deployed population.

*Balanced accuracy vs macro-F1?*
- BA = macro-recall → measures coverage of each class only.
- Macro-F1 folds in per-class precision → penalizes over-predicting a rare class.
```

&nbsp;

#### Top-k Accuracy
- **What**: Fraction of samples whose true class is among the $k$ highest-scored classes.
- **Why**: Large $K$ with genuinely ambiguous or hierarchical labels → top-1 prices a near-miss like a total miss.
- **How**: Sort the score vector → check membership of the true class in the top $k$.

```{note} Math
:class: dropdown
Definition:

$$
\text{Acc}@k=\frac{1}{m}\sum_{i=1}^{m}\mathbb{1}\left[y_i\in\text{top-}k(\hat{\mathbf{p}}_i)\right]
$$

Properties:
- Monotone non-decreasing in $k$.
- $k=1$ → accuracy. $k=K$ → $1$.
- Depends only on the score **ranking**, not on the probability values.
```

```{attention} Q&A
:class: dropdown
*When is it honest?*
- $k\ll K$, and the downstream consumer really does get $k$ shots (a human picks from a shortlist, a downstream filter reranks).

*When is it not?*
- Comparing across problems with different $K$ — top-5 of 10 classes and top-5 of 1000 classes are not the same task.
- When only one prediction can be acted on → it flatters the model.

*Why does it hide miscalibration?*
- Ranking-only → any monotone rescaling of the scores leaves it unchanged, same as AUC.
```

&nbsp;

### Precision
- **What**: Fraction of predicted positives that are correct.
- **Why**: Acting on a positive prediction costs something → need to know how often the alarm is false.
- **How**: TP ÷ everything flagged positive.

```{note} Math
:class: dropdown
Definition:

$$
P=\frac{\text{TP}}{\text{TP}+\text{FP}}
$$

Dependence on prevalence (Bayes):

$$
P=\frac{\text{TPR}\cdot\pi}{\text{TPR}\cdot\pi+\text{FPR}\cdot(1-\pi)}
$$

Properties:
- Undefined when nothing is predicted positive → convention $P=0$.
- Ignores TN and FN.
- $\pi\downarrow$ with TPR, FPR fixed → $P\downarrow$.
```

```{attention} Q&A
:class: dropdown
*When to prioritize it?*
- FP is expensive or irreversible: spam filtering (a real email is lost), automated content removal, recommending, a costly manual follow-up per alert.

*Why is it trivially gameable?*
- Predict positive only for the single highest-scored sample → $P\approx1$, $R\approx0$.
- → Never report precision without recall (or a fixed operating point).

*Why does it collapse at low prevalence?*
- Base rate fallacy: negatives outnumber positives, so even a tiny FPR generates more FPs than there are true positives.
- $\pi=0.001$, TPR$=0.99$, FPR$=0.01$ → $P=\frac{0.00099}{0.00099+0.00999}\approx9\%$. A "99% accurate test" is wrong 9 times out of 10 when it fires.

*Is it monotone in $\tau$?*
- ❌. Raising $\tau$ usually raises precision but can lower it — the sample dropped may have been a TP. This is exactly why the PR curve is jagged while the ROC curve is not.
```

&nbsp;

### Recall
- **What**: Fraction of actual positives that are caught.
- **Why**: Missing a positive costs something → need to know how many were never flagged.
- **How**: TP ÷ everything that is truly positive.

```{note} Math
:class: dropdown
Definition:

$$
R=\text{TPR}=\frac{\text{TP}}{\text{TP}+\text{FN}}
$$

Properties:
- Denominator is fixed by the data → invariant to $\pi$ and to how many negatives exist.
- Monotone non-increasing in $\tau$.
- Undefined when there are no positives.
```

```{attention} Q&A
:class: dropdown
*When to prioritize it?*
- FN is expensive: disease screening, fraud, security triage, retrieval feeding a cheap human/model reranker.

*Why is it trivially gameable?*
- Predict everything positive → $R=1$, $P=\pi$.

*Precision-recall tradeoff mechanism?*
- $\tau\downarrow$ → the predicted-positive set only grows → TP⬆️ & FP⬆️ → $R$⬆️ monotonically, $P$⬇️ on average.
- The tradeoff is a property of the **score ranking**: with a perfect ranking, both can be high simultaneously.

*Why do medical tests report sensitivity & specificity instead of precision?*
- Both condition on the truth → they transfer across populations with different $\pi$.
- Precision does not → the same test has a different PPV in a screening population and a symptomatic one.
```

&nbsp;

#### Specificity
- **What**: Fraction of actual negatives correctly rejected (TNR).
- **Why**: Recall says nothing about negatives → a model that flags everything is perfect by recall alone.
- **How**: TN ÷ everything that is truly negative.

```{note} Math
:class: dropdown
Definition:

$$
\text{TNR}=\frac{\text{TN}}{\text{TN}+\text{FP}}=1-\text{FPR}
$$

Properties:
- Recall of the negative class → the class-swapped mirror of TPR.
- Invariant to $\pi$.
- $(\text{TPR},1-\text{TNR})$ is exactly the point plotted on the ROC curve.
```

```{attention} Q&A
:class: dropdown
*Specificity vs precision?*
- Specificity: of all negatives, how many survived. Denominator = all negatives → stable.
- Precision: of all alarms, how many were real. Denominator = model output → collapses when negatives dominate.
- High specificity + low prevalence → still terrible precision.

*Where does it appear implicitly?*
- ROC's x-axis ($1-$TNR), balanced accuracy, Youden's $J=\text{TPR}+\text{TNR}-1$.

*Why is it missing from F1?*
- F1 is built from TP, FP, FN only → TN never enters → F1 is blind to how well negatives are handled, and changes if you swap which class is "positive".
```

&nbsp;

### F1
- **What**: Harmonic mean of precision & recall.
- **Why**: $P$ and $R$ are each trivially gameable and move in opposite directions with $\tau$ → need one number that no degenerate classifier can win.
- **How**: Harmonic mean → the score is dragged down to the smaller of the two.

```{note} Math
:class: dropdown
Definition:

$$
F_1=\frac{2PR}{P+R}=\frac{2\text{TP}}{2\text{TP}+\text{FP}+\text{FN}}
$$

Properties:
- Range $[0,1]$; convention $F_1=0$ when $\text{TP}=0$.
- $\min(P,R)\le F_1\le\frac{P+R}{2}$, with equality iff $P=R$.
- ❌TN → asymmetric under swapping the positive class.
- Non-decomposable → must be computed on pooled predictions.
```

````{important} Code
:class: dropdown
```python
import numpy as np

def counts(C):
    ## per-class one-vs-rest counts from a KxK confusion matrix (rows=true, cols=pred)
    tp = np.diag(C).astype(float)
    fp = C.sum(0) - tp
    fn = C.sum(1) - tp
    return tp, fp, fn

def fbeta(C, beta=1.0):
    tp, fp, fn = counts(C)
    ## (1+b^2)TP / ((1+b^2)TP + b^2 FN + FP) -- the count form, no 0/0 on P and R
    b2 = beta ** 2
    return (1 + b2) * tp / ((1 + b2) * tp + b2 * fn + fp)

## Example: class 1 is over-predicted -> high recall, poor precision
C = np.array([[80, 15],
              [ 2,  3]])
print(fbeta(C).round(3))              ## [0.904 0.261]
print(fbeta(C, beta=2).round(3))      ## [0.866 0.395]  <- recall weighted 2x helps class 1
```
````

```{attention} Q&A
:class: dropdown
*Why the harmonic mean?*
- Arithmetic mean of $(P,R)=(0.01,1.0)$ is $\approx0.5$ → the predict-everything-positive model looks average.
- Harmonic mean gives $0.02$ → dominated by the smaller term, so both must be high.
- $\text{HM}\le\text{GM}\le\text{AM}$ always.

*Cons?*
- ❌TN → relabeling which class is positive changes the score; a model can have $F_1=0.9$ on positives and be useless on negatives.
- Assumes $P$ and $R$ are equally important → use $F_\beta$ when they are not.
- Threshold-dependent → an "F1 of 0.7" without the threshold is not reproducible.
- Not chance-corrected → no fixed baseline to compare against, unlike MCC or $\kappa$.

*F1 vs accuracy?*
- Imbalanced + minority class is the point → F1 (or macro-F1).
- Balanced + symmetric costs → accuracy, which at least uses all four cells.

*How do you actually maximize it?*
- Train with a surrogate loss, then sweep $\tau$ on validation and keep the $\arg\max$ of F1.
- The F1-optimal threshold is generally **not** $0.5$; for a calibrated model it sits near $\frac{F_1^*}{2}$.
```

&nbsp;

#### F-beta
- **What**: Weighted harmonic mean where recall counts $\beta$ times as much as precision.
- **Why**: F1 hard-codes equal weight, but FP and FN almost never cost the same.
- **How**: Multiply recall's weight in the harmonic mean by $\beta^2$.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $\beta>0$: Relative weight of recall against precision.

Definition:

$$
F_\beta=(1+\beta^2)\frac{PR}{\beta^2P+R}=\frac{(1+\beta^2)\text{TP}}{(1+\beta^2)\text{TP}+\beta^2\text{FN}+\text{FP}}
$$

Properties:
- $\beta=1$ → $F_1$.
- $\beta\to0$ → $P$. $\beta\to\infty$ → $R$.
- $\beta>1$ → recall-leaning ($F_2$). $\beta<1$ → precision-leaning ($F_{0.5}$).
```

```{attention} Q&A
:class: dropdown
*How do you pick $\beta$?*
- From the $P$/$R$ balance you want: $F_\beta$ is indifferent between them exactly at $R=\beta P$.
- ❌ NOT an FP-per-FN exchange rate — that rate is $\frac{\text{FP}+\beta^2(\text{TP}+\text{FN})}{\text{TP}}$, which moves with the operating point.
- Screening (a missed case is fatal, a false alarm costs a second test) → $F_2$.
- Auto-moderation (a wrongly deleted post is expensive) → $F_{0.5}$.

*Why $\beta^2$ and not $\beta$?*
- $\frac{\partial F_\beta}{\partial P}=\frac{(1+\beta^2)R^2}{(\beta^2P+R)^2}$ and $\frac{\partial F_\beta}{\partial R}=\frac{(1+\beta^2)\beta^2P^2}{(\beta^2P+R)^2}$ → they are equal exactly at $R=\beta P$.
- → The squaring is what makes the plain reading ("recall is $\beta$ times as important") literally true.

*Why not just use expected cost?*
- You should, whenever the cost **ratio** is known — $\tau^*$ needs nothing more than that ratio.
- $F_\beta$ is the fallback when TN is not even countable (unbounded negative pools) → a per-sample expected cost is undefined.
```

&nbsp;

### Averaging
- **What**: Rule for collapsing $K$ per-class scores into one number.
- **Why**: P/R/F1 are binary by construction → $K>2$ or multi-label produces $K$ scores with no defined ordering.
- **How**:
    - **Micro**: pool TP/FP/FN across classes → compute the metric once.
    - **Macro**: compute per class → unweighted mean.
    - **Weighted**: compute per class → mean weighted by support.
    - **Samples**: compute per sample over its label set → mean (multi-label only).

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $m_k$: #samples of true class $k$ (support).

Definitions:

$$\begin{align*}
P_\text{micro}&=\frac{\sum_k\text{TP}_k}{\sum_k(\text{TP}_k+\text{FP}_k)}\\
P_\text{macro}&=\frac{1}{K}\sum_kP_k\\
P_\text{weighted}&=\sum_k\frac{m_k}{m}P_k
\end{align*}$$

Properties:
- Single-label multi-class with all $K$ classes included: each error is one FP and one FN → $\sum_k\text{FP}_k=\sum_k\text{FN}_k$ → $P_\text{micro}=R_\text{micro}=F_{1,\text{micro}}=\text{Acc}$.
- Macro-recall $=$ balanced accuracy.
- Macro-F1 $\neq$ the F1 of macro-$P$ and macro-$R$ ← the harmonic mean does not commute with averaging.
```

````{important} Code
:class: dropdown
```python
import numpy as np

## Reuses counts() from the F1 block above
def counts(C):
    tp = np.diag(C).astype(float)
    return tp, C.sum(0) - tp, C.sum(1) - tp

def micro_f1(C):
    tp, fp, fn = counts(C)
    ## pool the counts FIRST, then compute a single F1
    return 2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum())

def macro_f1(C):
    tp, fp, fn = counts(C)
    ## per-class F1 FIRST, then an unweighted mean -> the rare class gets a full vote
    return (2 * tp / (2 * tp + fp + fn)).mean()

## Example: class 2 has 5 samples and is never predicted
C = np.array([[50,  0, 0],
              [ 0, 45, 0],
              [ 3,  2, 0]])
print(round(micro_f1(C), 3), round(macro_f1(C), 3))   ## 0.95 0.65
```
````

```{attention} Q&A
:class: dropdown
*Micro vs macro in one line?*
- Micro weights every **sample** equally → the frequent classes decide the score.
- Macro weights every **class** equally → totally failing one rare class costs $\frac{1}{K}$ of the score.

*Which to report on imbalanced data?*
- Macro (plus the per-class table). Micro-F1 on a single-label problem is just accuracy under a different name, so it inherits the accuracy paradox.

*Weighted-F1 gotcha?*
- Weighting by support re-introduces majority dominance → it tracks accuracy closely and mostly defeats the purpose of macro averaging.
- It is also not bounded between weighted-$P$ and weighted-$R$.

*Why can't you average F1 across CV folds?*
- F1 is non-decomposable → the mean of per-fold F1s $\neq$ the F1 of pooled out-of-fold predictions, and the gap grows as folds get smaller / more imbalanced.

*Multi-label: micro vs samples?*
- Micro pools over all (sample, label) pairs → dominated by frequent labels.
- Samples computes an F1 per sample over its own label set → measures per-item set quality.
```

&nbsp;

### ROC-AUC
- **Name**: Area Under the Receiver Operating Characteristic Curve
- **What**: Probability that a random positive is scored above a random negative.
- **Why**: Every count metric is tied to one threshold → comparing models or reporting before the operating point is chosen needs a threshold-free number.
- **How**:
    1. Sort samples by score, descending.
    2. Sweep $\tau$ from $+\infty$ to $-\infty$, plotting $(\text{FPR}(\tau),\text{TPR}(\tau))$.
    3. Integrate under the resulting staircase.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $m_+$: #positives.
    - $m_-$: #negatives.
    - $s^+$: Score of a uniformly random positive.
    - $s^-$: Score of a uniformly random negative.
    - $r_i$: Midrank of sample $i$ when all scores are sorted ascending.

Curve:

$$
\text{ROC}=\left\{\left(\text{FPR}(\tau),\text{TPR}(\tau)\right):\tau\in\mathbb{R}\right\}
$$

Probabilistic identity:

$$
\text{AUC}=\int_0^1\text{TPR}\,d(\text{FPR})=P(s^+>s^-)+\frac{1}{2}P(s^+=s^-)
$$

Rank form (normalized Mann-Whitney $U$):

$$
\text{AUC}=\frac{\sum_{i:y_i=1}r_i-\frac{m_+(m_++1)}{2}}{m_+m_-}
$$

Properties:
- $0.5$ = random, $1$ = perfect, $<0.5$ = systematically inverted (flip the sign of $s$).
- Invariant to any strictly increasing transform of $s$ → ❌calibration signal.
- Invariant to $\pi$ ← TPR uses only positives, FPR only negatives.
- Gini coefficient $=2\text{AUC}-1$.
```

````{important} Code
:class: dropdown
```python
import numpy as np

def roc_auc(y, s):
    ## rank-based (Mann-Whitney U) -- no threshold sweep needed
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    ## midranks: tied scores share the average rank -> ties count as 0.5
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    ranks = (np.bincount(inv, weights=ranks) / cnt)[inv]
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

## Example: 1 of the 4 positive/negative pairs is mis-ordered
y = np.array([0, 0, 1, 1])
s = np.array([0.1, 0.4, 0.35, 0.8])
print(roc_auc(y, s))   ## 0.75
```
````

```{attention} Q&A
:class: dropdown
*What does an AUC of 0.8 actually mean?*
- Pick one random positive and one random negative → the positive gets the higher score $80\%$ of the time.
- It says nothing about how many mistakes the deployed model makes, because no threshold has been chosen.

*Pros?*
- Threshold-free → compare models before committing to an operating point.
- $\pi$-invariant → stable across test sets with different class balance, and across resampled/downsampled negatives.
- Has a fixed, universal baseline of $0.5$.

*Cons?*
- ❌calibration → a model with AUC $=1$ can output probabilities that are all wrong.
- Averages over the whole curve, including regions (e.g., FPR $=0.9$) you would never deploy at → use partial AUC when only a region matters.
- Optimistic under extreme imbalance: thousands of FPs barely move FPR because the denominator is all negatives, while precision is destroyed.

*Multi-class?*
- OvR: AUC per class vs the rest, macro/weighted averaged. Sensitive to $\pi$ through the "rest" class.
- OvO (Hand-Till): average AUC over all class pairs. $\pi$-insensitive but $O(K^2)$.

*How do you pick the operating threshold from it?*
- Max Youden's $J=\text{TPR}-\text{FPR}$ (the one-sided KS statistic) → the point furthest above the diagonal.
- Or the cost-optimal $\tau^*$ if costs are known; or the $\tau$ hitting a required TPR/FPR.
- Always on validation data.

*Is AUC $=0.5$ proof the model is useless?*
- ❌. It proves the scores are not monotonically related to the label. A U-shaped relation is highly informative and still gives $0.5$.
```

&nbsp;

### PR-AUC
- **Name**: Area Under the Precision-Recall Curve
- **What**: Threshold-free summary of precision across all recall levels.
- **Why**: ROC-AUC hides FP explosions under extreme imbalance ← FPR's denominator is dominated by negatives, so precision can collapse while FPR barely moves.
- **How**:
    1. Sort by score, descending.
    2. Walk down the list, recomputing $(R,P)$ after each sample.
    3. Sum $P$ weighted by the recall increments (Average Precision).

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $P_n$: Precision after taking the top $n$ scored samples.
    - $R_n$: Recall after taking the top $n$ scored samples.

Average Precision (step-wise, no interpolation):

$$
\text{AP}=\sum_n(R_n-R_{n-1})P_n
$$

Properties:
- No-skill baseline (the flat PR line at precision $\pi$) $=\pi$, NOT $0.5$ → the score is not comparable across datasets with different $\pi$.
- ❌TN → invisible to how many negatives were correctly ignored.
- The PR curve between two achievable points is a hyperbola, not a line → linear interpolation passes through operating points no threshold attains; trapezoidal `auc(recall, precision)` is invalid and usually (not always) optimistic.
- Perfect ranking → $\text{AP}=1$; the curve's left end ($R\to0$) is estimated from a handful of samples → high variance.
```

````{important} Code
:class: dropdown
```python
import numpy as np

def average_precision(y, s):
    ## assumes distinct scores; libraries group tied scores into one threshold
    y = y[np.argsort(-s, kind="mergesort")]      ## highest score first
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)    ## P at every cut-off
    ## recall jumps by 1/n_pos exactly at a positive -> mask precision by y
    return float((precision * y).sum() / y.sum())

## Example: positives ranked 1st and 3rd
y = np.array([0, 1, 0, 1])
s = np.array([0.1, 0.9, 0.4, 0.35])
print(round(average_precision(y, s), 3))   ## 0.833
```
````

```{dropdown} Table: ROC-AUC vs PR-AUC
| | ROC-AUC | PR-AUC (AP) |
|:--|:--|:--|
| Axes | TPR vs FPR | Precision vs Recall |
| Uses TN | ✅ | ❌ |
| Random baseline | $0.5$ | $\pi$ |
| Sensitive to $\pi$ | ❌ | ✅ |
| Comparable across test sets | ✅ | ❌ (only at equal $\pi$) |
| Behavior at $\pi\to0$ | Stays optimistic | Drops with precision |
| Use when | Ranking quality, balanced-ish data, negatives matter | Rare positives, the cost is in the FPs you act on |
```

```{attention} Q&A
:class: dropdown
*Why is PR-AUC preferred under heavy imbalance?*
- Precision's denominator is what the model flagged, so every extra FP hurts immediately.
- FPR's denominator is the (huge) negative pool, so the same FPs are diluted → ROC stays flattering.

*AP vs "area under the PR curve"?*
- AP = the step-wise sum above, and is what `average_precision_score` reports.
- Trapezoidal integration of the PR curve interpolates through points no threshold achieves → not reproducible across libraries, and it can move the number in either direction. Report AP.

*Why can't PR-AUC be compared across datasets?*
- Its floor is $\pi$. AP $=0.4$ at $\pi=0.01$ is excellent; at $\pi=0.5$ it is worse than random.
- Fix: report AP alongside $\pi$, or normalize as $\frac{\text{AP}-\pi}{1-\pi}$.

*Which curve is monotone?*
- ROC is monotone non-decreasing by construction. The PR curve is not — precision can rise and fall as $\tau$ drops → sawtooth.

*Does a model that dominates in ROC space dominate in PR space?*
- ✅ Yes, and vice versa: a curve dominating in one space dominates in the other. Their *summary areas* can still rank two crossing models differently.
```

&nbsp;

### MCC
- **Name**: Matthews Correlation Coefficient
- **What**: Pearson correlation between the true and predicted binary labels.
- **Why**: Accuracy is fooled by imbalance and F1 ignores TN → both can look strong for a classifier that has learned only the majority class.
- **How**: Treat the two 0/1 label vectors as variables → correlate them → all four cells enter symmetrically.

```{note} Math
:class: dropdown
Definition:

$$
\text{MCC}=\frac{\text{TP}\cdot\text{TN}-\text{FP}\cdot\text{FN}}{\sqrt{(\text{TP}+\text{FP})(\text{TP}+\text{FN})(\text{TN}+\text{FP})(\text{TN}+\text{FN})}}
$$

Properties:
- Range $[-1,1]$: $1$ perfect, $0$ random/degenerate, $-1$ perfectly inverted.
- Equals the $\phi$ coefficient, i.e., Pearson $r$ between the 0/1 vectors $\mathbf{y}$ and $\hat{\mathbf{y}}$.
- Symmetric under swapping the positive class, and under swapping $y\leftrightarrow\hat{y}$.
- Any constant predictor → a zero factor in the denominator → convention $\text{MCC}=0$.
- $\text{FP}=\text{FN}$ with nondegenerate marginals → $\text{MCC}=\kappa$ exactly.
```

```{attention} Q&A
:class: dropdown
*Pros?*
- Uses all four cells → high MCC requires doing well on **both** classes, whatever $\pi$ is.
- Chance-corrected with a fixed $0$ baseline → interpretable without knowing $\pi$.
- Class-swap symmetric → no arbitrary choice of which class is "positive".

*Cons?*
- Still a single-threshold metric.
- Less intuitive than P/R → hard to translate into a product decision.
- Degenerate confusion matrices force a convention rather than a value.

*MCC vs F1?*
- F1: ignores TN, asymmetric, no fixed baseline; suits retrieval-style problems with an unbounded negative pool.
- MCC: uses TN, symmetric, chance-corrected; the safer default for imbalanced binary classification with a well-defined negative class.

*MCC vs balanced accuracy?*
- BA averages the two recalls → ignores precision.
- MCC also reacts to the FP count → a model that boosts minority recall by flooding FPs scores high on BA and low on MCC.
```

&nbsp;

### Cohen's Kappa
- **What**: Agreement between two labelings, corrected for the agreement expected by chance.
- **Why**: Raw agreement is inflated by skewed marginals → two raters who both mostly say "no" agree $90\%$ of the time with zero skill.
- **How**:
    1. Observed agreement = diagonal mass.
    2. Expected agreement = what independent raters with the same marginals would hit.
    3. Normalize the gap by the remaining headroom.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $p_o$: Observed agreement.
    - $p_e$: Chance agreement.
    - $C_{k\cdot},C_{\cdot k}$: Row / column sums of the confusion matrix.

Definition:

$$
\kappa=\frac{p_o-p_e}{1-p_e},\qquad p_o=\frac{1}{m}\sum_kC_{kk},\qquad p_e=\frac{1}{m^2}\sum_kC_{k\cdot}C_{\cdot k}
$$

Weighted form (ordinal labels):

$$
\kappa_w=1-\frac{\sum_{k,l}w_{kl}C_{kl}}{\sum_{k,l}w_{kl}E_{kl}}
$$
- $w_{kl}$: Disagreement cost, linear $|k-l|$ or quadratic $(k-l)^2$.
- $E_{kl}=\frac{C_{k\cdot}C_{\cdot l}}{m}$: Expected count under independence.

Properties:
- $\kappa\le1$; $\kappa=0$ at chance; $\kappa<0$ for worse-than-chance agreement.
- $p_e$ depends on both marginals → $\kappa$ is not comparable across datasets.
```

```{attention} Q&A
:class: dropdown
*Where is it actually used?*
- Inter-annotator agreement → whether the labels themselves are trustworthy. Low $\kappa$ → the task is ill-defined or the labels are noisy → every metric measured against them is unreliable.
- Ordinal targets (quadratic weighted kappa) → severity grades, star ratings, essay scores.

*Kappa paradox?*
- Highly skewed marginals inflate $p_e$ → $p_e\to1$ makes the denominator vanish → $\kappa$ can be near $0$ despite $p_o=0.95$.
- → Never report $\kappa$ without $p_o$ and the marginals.

*$\kappa$ vs MCC?*
- Both chance-correct a confusion matrix; $\kappa$ subtracts expected *agreement*, MCC is a *correlation*.
- Binary with $\text{FP}=\text{FN}$ → identical. Otherwise MCC is the more stable choice for classifier evaluation, $\kappa$ the standard for rater agreement.

*Why linear vs quadratic weights?*
- Linear → error cost grows with distance.
- Quadratic → large disagreements dominate; standard for ordinal ML competitions, and equivalent to a chance-corrected MSE on the label indices.
```

&nbsp;

## Probabilistic
### Log Loss
- **What**: Negative log-probability assigned to the true label.
- **Why**: Hard-label metrics score a $0.51$ and a $0.99$ prediction identically → no signal about confidence, which is what a downstream decision rule consumes.
- **How**: Take the predicted probability of the true class → $-\log$ → average.

```{note} Math
:class: dropdown
Definition:

$$
\text{LL}=-\frac{1}{m}\sum_{i=1}^{m}\sum_{k=1}^{K}y_{ik}\log\hat{p}_{ik}
$$
- $y_{ik}\in\{0,1\}$: One-hot true label.

Properties:
- Identical in form to the [cross entropy](obj.md#cross-entropy) training loss → the one metric you can also optimize directly.
- **Strictly proper**: uniquely minimized in expectation by $\hat{p}=P(y|\mathbf{x})$.
- Unbounded above: one confident mistake ($\hat{p}\to0$) sends it to $\infty$ → probabilities are clipped to $[\epsilon,1-\epsilon]$.
- Baseline (predict the marginal class distribution) $=H(y)$; for binary, $H(\pi)$.
- Decomposable per sample.
```

```{attention} Q&A
:class: dropdown
*What is a proper scoring rule and why care?*
- A score whose expectation is minimized only by reporting the true probability → no incentive to shade predictions toward $0/1$ to game the number.
- Log loss & Brier are strictly proper. Accuracy, F1, AUC, ECE are not.
- Counterexample: mean absolute error on probabilities is improper — $\mathbb{E}|q-y|$ is linear in $q$ → minimized by reporting exactly $0$ or $1$.

*Pros?*
- Scores discrimination **and** calibration in one number.
- Decomposable → usable as a training loss and on minibatches.
- Interpretable against a real baseline: $\text{LL}<H(\pi)$ means the features carry information.

*Cons?*
- Unbounded → a handful of confident errors dominate the average, and the reported value depends on the clipping $\epsilon$.
- Not interpretable in absolute terms; the scale moves with $K$ and $\pi$.
- Requires probabilities, not arbitrary scores.

*Log loss vs AUC?*
- AUC only sees the ranking; log loss sees the values.
- A monotone rescaling of the scores leaves AUC untouched and can change log loss arbitrarily → they routinely disagree, and improving one can worsen the other.
```

&nbsp;

### Brier Score
- **What**: Mean squared error between predicted probabilities and one-hot labels.
- **Why**: Log loss is unbounded → a single overconfident mistake on a hard sample can dominate the whole report.
- **How**: Squared difference per class → sum over classes → average over samples.

```{note} Math
:class: dropdown
Definition (multi-class, sum form):

$$
\text{BS}=\frac{1}{m}\sum_{i=1}^{m}\sum_{k=1}^{K}(\hat{p}_{ik}-y_{ik})^2
$$

Definition (binary convention):

$$
\text{BS}_2=\frac{1}{m}\sum_{i=1}^{m}(s_i-y_i)^2=\frac{1}{2}\left.\text{BS}\right|_{K=2}
$$
- The two-class sum double-counts each error ← $(\hat{p}_{i0}-y_{i0})^2=(\hat{p}_{i1}-y_{i1})^2$.

Murphy decomposition (binary, forecasts grouped by **distinct value**):

$$
\text{BS}_2=\underbrace{\text{REL}}_{\text{calibration gap}}-\underbrace{\text{RES}}_{\text{discrimination}}+\underbrace{\text{UNC}}_{\pi(1-\pi)}
$$
- Exact only when $s$ takes finitely many distinct values; binning genuinely different forecasts together leaves a within-bin remainder.

Properties:
- Strictly proper.
- Bounded: $[0,1]$ for $\text{BS}_2$, $[0,2]$ for the multi-class sum form.
- Baseline (constant $\hat{p}=\pi$) $=\pi(1-\pi)$, which is exactly UNC.
- Quadratic penalty → far gentler than $-\log$ on confident errors.
```

```{attention} Q&A
:class: dropdown
*Brier vs log loss?*
- Both strictly proper. Brier is bounded and robust to a few overconfident mistakes; log loss punishes them without limit.
- Confident errors are catastrophic in the application (medicine, safety) → log loss. A few noisy labels are expected → Brier.

*How do you read the decomposition?*
- REL$\downarrow$ = better calibrated. RES$\uparrow$ = predictions separate the classes. UNC is fixed by the data and cannot be improved.
- → Two models with the same Brier can differ completely in *why*.

*Baseline?*
- $\pi(1-\pi)$, maximized at $0.25$ when $\pi=0.5$ → for rare events the baseline is tiny, so a raw Brier of $0.01$ can still be worthless.
- Fix: Brier skill score $=1-\frac{\text{BS}}{\text{BS}_\text{ref}}$.

*Does a good Brier imply good calibration?*
- ❌. Brier trades REL against RES → a sharp, slightly miscalibrated model can beat a calibrated, uninformative one. Report ECE alongside it.
```

&nbsp;

### ECE
- **Name**: Expected Calibration Error
- **What**: Average gap between predicted confidence and observed accuracy, binned by confidence.
- **Why**: Proper scores blend calibration with discrimination → they cannot answer "when the model says $90\%$, is it right $90\%$ of the time?", which is what any cost-based threshold assumes.
- **How**:
    1. Bin samples by predicted confidence.
    2. Per bin: |accuracy $-$ mean confidence|.
    3. Weight by bin size → sum.

```{note} Math
:class: dropdown
Notations:
- Hyperparams:
    - $B$: #bins.
- Misc:
    - $\mathcal{B}_b$: Set of samples whose confidence falls in bin $b$.
    - $\text{conf}(\mathcal{B}_b)=\frac{1}{|\mathcal{B}_b|}\sum_{i\in\mathcal{B}_b}\max_k\hat{p}_{ik}$: Mean top-label confidence.
    - $\text{acc}(\mathcal{B}_b)$: Fraction of the bin predicted correctly.

Definition:

$$
\text{ECE}=\sum_{b=1}^{B}\frac{|\mathcal{B}_b|}{m}\left|\text{acc}(\mathcal{B}_b)-\text{conf}(\mathcal{B}_b)\right|
$$

Worst-case variant:

$$
\text{MCE}=\max_{b:\mathcal{B}_b\neq\emptyset}\left|\text{acc}(\mathcal{B}_b)-\text{conf}(\mathcal{B}_b)\right|
$$

Properties:
- Top-label calibration $\Leftrightarrow$ $P(y=\hat{y}\mid\text{conf}=p)=p$ for all $p$ → $\text{ECE}=0$. Strictly weaker than full multi-class calibration, which constrains all $K$ probabilities.
- ❌proper scoring rule → minimizable without any predictive skill.
- Only the top-label probability is checked; classwise ECE averages the same quantity over all $K$ columns.
```

````{important} Code
:class: dropdown
```python
import numpy as np

def ece(y, probs, n_bins=10):
    conf = probs.max(1)                   ## top-label confidence
    correct = (probs.argmax(1) == y).astype(float)
    ## equal-width bins over [0, 1]; bin b covers ((b-1)/B, b/B]
    b = np.clip(np.ceil(conf * n_bins).astype(int) - 1, 0, n_bins - 1)
    e = 0.0
    for k in range(n_bins):
        mask = b == k
        if mask.any():
            ## weight each bin's |accuracy - confidence| gap by how full it is
            e += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return e

## Example: always 90% confident, only right half the time
y = np.array([0, 1, 0, 1])
probs = np.array([[0.9, 0.1]] * 4)
print(round(ece(y, probs), 3))   ## 0.4
```
````

```{attention} Q&A
:class: dropdown
*Why is it not a proper scoring rule?*
- The constant predictor $\hat{p}=\pi$ for every sample is perfectly calibrated → $\text{ECE}\approx0$ with zero discrimination.
- → ECE is only meaningful **alongside** log loss/Brier/AUC, never alone.

*Cons?*
- Binning-dependent: $B$ and the scheme (equal-width vs equal-mass/adaptive) change the number.
- Biased estimator: sampling noise inside a bin inflates $|\text{acc}-\text{conf}|$, while a bin mixing over- and under-confident samples cancels real error out.
- Top-label only → says nothing about the other $K-1$ probabilities.
- Depends only on per-bin aggregates → a model that ranks perfectly and one that ranks randomly can score the same ECE.

*How do you fix a miscalibrated model?*
- **Temperature scaling**: one scalar $T$ on the logits, fit on validation NLL. Preserves the argmax → accuracy unchanged, binary AUC unchanged. Multi-class OvR AUC CAN move ← softmax renormalization is not monotone in an individual class's probability. Default first choice.
- **Platt scaling**: logistic regression on the score (2 params) → handles a shift as well as a scale.
- **Isotonic regression**: non-parametric monotone fit → strictly more flexible, needs far more validation data, and can overfit into a staircase.

*Why are modern NNs overconfident?*
- Trained to near-zero loss against one-hot targets with capacity to spare → the logit gap keeps growing after the argmax is already correct.
- Mitigations: [label smoothing](obj.md#label-smoothing), weight decay, mixup, and post-hoc temperature scaling.

*Reliability diagram?*
- Plot $\text{acc}(\mathcal{B}_b)$ against $\text{conf}(\mathcal{B}_b)$; the diagonal is perfect. Below → overconfident, above → underconfident. ECE is the bin-size-weighted $L_1$ area between the curve and the diagonal.
```

&nbsp;

## Regression
### RMSE
- **Name**: Root Mean Squared Error
- **What**: Square root of the mean squared residual.
- **Why**: [MSE](obj.md#mse) lives in squared target units → not readable next to the target and not comparable with MAE.
- **How**: MSE → square root.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $e_i=y_i-\hat{y}_i$: Residual of sample $i$.
    - $\mathbf{e}\in\mathbb{R}^m$: Vector of all residuals.
    - $\bar{e}$: Mean residual.

Definition:

$$
\text{RMSE}=\sqrt{\frac{1}{m}\sum_{i=1}^{m}(y_i-\hat{y}_i)^2}
$$

Properties:
- Same units as $y$.
- $\text{MAE}\le\text{RMSE}\le\sqrt{m}\cdot\text{MAE}$ ← $\|\mathbf{e}\|_2\le\|\mathbf{e}\|_1\le\sqrt{m}\|\mathbf{e}\|_2$. Left equality iff all residuals are equal in magnitude; right iff at most one is nonzero.
- $\text{RMSE}^2=\bar{e}^2+\text{Var}(e)$ → decomposes the reported error into systematic bias and spread.
- Minimized by the conditional mean $\mathbb{E}[y|\mathbf{x}]$.
- Scale-dependent → meaningless across differently-scaled targets.
```

```{attention} Q&A
:class: dropdown
*RMSE or MSE?*
- Monotone in each other → identical model ranking. RMSE is for reporting (units), MSE for training (smooth gradient, no $\frac{1}{2\sqrt{\cdot}}$ factor).

*RMSE vs MAE?*
- RMSE weights each residual by its own magnitude → a few large errors dominate.
- $\frac{\text{RMSE}}{\text{MAE}}$ is a free outlier diagnostic: $\approx1$ → uniform errors; $\gg1$ → a heavy tail.
- Large errors are disproportionately costly (over-capacity, safety margins) → RMSE. Outliers are label noise → MAE.

*How to compare RMSE across datasets?*
- You cannot directly. Normalize: $\frac{\text{RMSE}}{\bar{y}}$ or $\frac{\text{RMSE}}{y_{\max}-y_{\min}}$, or report $R^2$, which divides by the variance of $y$.

*Why does RMSE reward predicting the mean?*
- Its population minimizer is $\mathbb{E}[y|\mathbf{x}]$ → on a skewed or multimodal conditional, the RMSE-optimal prediction is a value no sample ever takes.
```

&nbsp;

#### RMSLE
- **Name**: Root Mean Squared Log Error
- **What**: RMSE computed on $\log(1+y)$.
- **Why**: Targets spanning orders of magnitude → RMSE is decided entirely by the largest ones, while the business error is relative.
- **How**: $\log(1+\cdot)$-transform both sides → RMSE.

```{note} Math
:class: dropdown
Definition:

$$
\text{RMSLE}=\sqrt{\frac{1}{m}\sum_{i=1}^{m}\left(\log(1+\hat{y}_i)-\log(1+y_i)\right)^2}
$$

Properties:
- $\log(1+\hat{y})-\log(1+y)=\log\frac{1+\hat{y}}{1+y}$ → penalizes the **ratio**, not the difference.
- $\approx\frac{\hat{y}-y}{y}$ (relative error) when $y\gg1$ and the error is small.
- Requires $y,\hat{y}>-1$; the $+1$ exists to admit $y=0$.
- Asymmetric: at equal absolute error, under-prediction costs more ← $\log$ is concave.
```

```{attention} Q&A
:class: dropdown
*Concrete asymmetry?*
- $y=1000$: $\hat{y}=600$ → $|\log\frac{601}{1001}|=0.51$; $\hat{y}=1400$ → $|\log\frac{1401}{1001}|=0.34$.
- → Same $400$ absolute error, $1.5\times$ the penalty for under-shooting.

*When to use it?*
- Positive, long-right-tailed targets: demand, counts, prices, durations. Especially when a $2\times$ miss on a small value is as bad as a $2\times$ miss on a large one.

*Equivalent alternative?*
- Train on $\log(1+y)$ with plain MSE → the same objective. But then the reported error is in log space, and back-transforming $\mathbb{E}[\log y]$ under-estimates $\mathbb{E}[y]$ (Jensen).
```

&nbsp;

### MAE
- **Name**: Mean Absolute Error
- **What**: Mean absolute residual.
- **Why**: RMSE lets a handful of large residuals dictate the reported number, which misrepresents typical performance.
- **How**: |residual| → average.

```{note} Math
:class: dropdown
Definition:

$$
\text{MAE}=\frac{1}{m}\sum_{i=1}^{m}|y_i-\hat{y}_i|
$$

Properties:
- Same units as $y$; reads directly as "the typical miss".
- Every residual contributes linearly, not quadratically → outliers pull far less than under RMSE, though a large enough one still dominates.
- Minimized by the conditional median → gradient & MLE story in [MAE loss](obj.md#mae).
- Median Absolute Error $=\text{median}_i|e_i|$: $50\%$ breakdown point, ignores the tail entirely.
```

```{attention} Q&A
:class: dropdown
*Pros?*
- Robust to outliers and to heavy-tailed noise.
- The only common regression metric a non-technical stakeholder reads correctly on the first try.

*Cons?*
- Prices a $10\times$ error at $10\times$, not $100\times$ → wrong whenever the cost of an error is superlinear.
- Reports the median-optimal predictor → systematically "wrong" if the downstream use needs an unbiased **total** (summing medians $\neq$ the total).
- Scale-dependent, like RMSE.

*Non-differentiability at 0 — does it matter here?*
- ❌ for a metric: nothing is being differentiated. It only matters when MAE is used as the training loss.

*Which one do you report?*
- Both. MAE for the typical case, RMSE for the tail; their ratio tells you which regime you are in.
```

&nbsp;

### MAPE
- **Name**: Mean Absolute Percentage Error
- **What**: Mean absolute error as a fraction of the true value.
- **Why**: Absolute errors are not comparable across items/series with different scales → an error of $10$ is trivial on $10{,}000$ and fatal on $12$.
- **How**: |error| ÷ |truth| → average → $\times100\%$.

```{note} Math
:class: dropdown
Definition:

$$
\text{MAPE}=\frac{100\%}{m}\sum_{i=1}^{m}\frac{|y_i-\hat{y}_i|}{|y_i|}
$$

Properties:
- Scale-free → aggregates heterogeneous targets into one number.
- Undefined at $y_i=0$ and explodes as $y_i\to0$.
- Asymmetric for $y>0,\hat{y}\ge0$: under-prediction is capped at $100\%$ (at $\hat{y}=0$), over-prediction is unbounded.
- Population minimizer (positive targets) = the $\frac{1}{y}$-weighted median → at or **below** the ordinary median.
```

```{attention} Q&A
:class: dropdown
*Why does optimizing MAPE bias forecasts low?*
- The $\frac{1}{y}$ weight makes errors on small targets dominate, and over-prediction is penalized without limit while under-prediction is capped.
- → The minimizer is the $\frac{1}{y}$-weighted median, at or below the ordinary median → systematic under-forecasting.

*Cons?*
- Zero/near-zero truths → undefined or explosive.
- Asymmetric → not a fair comparison between a model that over-shoots and one that under-shoots.
- A single tiny $y_i$ can swamp the whole average.

*Fixes?*
- **WAPE** $=\frac{\sum|y_i-\hat{y}_i|}{\sum|y_i|}$: one division at the end → immune to individual zeros, and it is the metric most demand-forecasting stakeholders actually mean.
- **sMAPE**: bounded, but still asymmetric.
- **RMSLE**: relative in log space, defined at $y=0$.
```

&nbsp;

#### sMAPE
- **Name**: Symmetric Mean Absolute Percentage Error
- **What**: MAPE with the mean of $|y|$ and $|\hat{y}|$ in the denominator.
- **Why**: MAPE is undefined at $y=0$ and unbounded above → a single item can decide the score.
- **How**: Replace the denominator $|y|$ with $\frac{|y|+|\hat{y}|}{2}$.

```{note} Math
:class: dropdown
Definition:

$$
\text{sMAPE}=\frac{100\%}{m}\sum_{i=1}^{m}\frac{|y_i-\hat{y}_i|}{\frac{1}{2}(|y_i|+|\hat{y}_i|)}
$$

Properties:
- Bounded in $[0\%,200\%]$.
- Defined whenever $|y_i|+|\hat{y}_i|>0$.
- The denominator depends on the prediction → the metric is no longer a pure function of the error.
```

```{attention} Q&A
:class: dropdown
*Is it actually symmetric?*
- ❌, despite the name. $y=100$: $\hat{y}=150$ → $\frac{50}{125}=40\%$; $\hat{y}=50$ → $\frac{50}{75}=67\%$.
- → Under-prediction is still penalized harder; the "symmetry" only refers to the denominator's form.

*Other problems?*
- $200\%$ ceiling makes catastrophic misses indistinguishable from merely bad ones.
- $\hat{y}$ in the denominator → a model can lower sMAPE by inflating predictions, not by being more accurate.
- Multiple incompatible definitions in circulation (some drop the $\frac{1}{2}$, halving the range) → always state the formula.
```

&nbsp;

### R²
- **Name**: Coefficient of Determination
- **What**: Fraction of the target's variance explained, relative to predicting its mean.
- **Why**: RMSE is scale-dependent → "is an error of $3.2$ good?" has no answer without a baseline.
- **How**:
    1. Residual sum of squares of the model.
    2. Total sum of squares of the constant-mean predictor.
    3. $1-$ ratio.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $\bar{y}$: Mean of $y$ over the **evaluated** set.

Definition:

$$
R^2=1-\frac{\text{SS}_\text{res}}{\text{SS}_\text{tot}}=1-\frac{\sum_i(y_i-\hat{y}_i)^2}{\sum_i(y_i-\bar{y})^2}
$$

Properties:
- $R^2=1$ perfect; $R^2=0$ ties the mean predictor; $R^2<0$ worse than the mean — possible on held-out data.
- $R^2=1-\frac{\text{MSE}}{\text{Var}(y)}$ → a monotone rescaling of RMSE **within** one test set → identical model ranking there.
- $R^2=r^2$ (squared Pearson correlation between $y$ and $\hat{y}$) is **guaranteed** only for an in-sample OLS fit with an intercept ← the residuals are then zero-mean and orthogonal to $\hat{y}$.
- In-sample OLS: adding any feature cannot increase $\text{SS}_\text{res}$ → $R^2$ is monotone non-decreasing in $n$.
```

```{attention} Q&A
:class: dropdown
*Can it be negative?*
- ✅ on test/CV data, or for any model not fit by OLS-with-intercept on that same data. It just means you would have done better predicting $\bar{y}$.

*Why is it not comparable across datasets?*
- $\text{SS}_\text{tot}$ is the variance of $y$ **in the evaluated set** → a low-variance test set makes any model look bad, and an artificially diverse one flatters it.
- → Two test sets, same model, different $R^2$.

*$R^2$ vs correlation?*
- $r$ is invariant to affine rescaling of $\hat{y}$: $\hat{y}=2y+5$ gives $r=1$ but a badly negative $R^2$.
- $R^2$ measures agreement on the identity line; $r$ only measures co-movement.

*Explained variance vs $R^2$?*
- $\text{EV}=1-\frac{\text{Var}(y-\hat{y})}{\text{Var}(y)}$ subtracts the mean residual → blind to a constant bias.
- Equal iff the residuals have zero mean; $\text{EV}>R^2$ whenever the model is biased.

*Does a high $R^2$ mean the model is good?*
- ❌. It means the model beats the mean on this sample. It says nothing about causality, extrapolation, or whether the residuals are structured — always look at residual plots too.
```

&nbsp;

#### Adjusted R²
- **What**: $R^2$ penalized for the number of features.
- **Why**: In-sample $R^2$ never decreases when a feature is added, even pure noise → it cannot be used to compare models of different size.
- **How**: Rescale the residual and total sums of squares by their degrees of freedom before taking the ratio.

```{note} Math
:class: dropdown
Definition:

$$
R^2_\text{adj}=1-(1-R^2)\frac{m-1}{m-n-1}
$$

Properties:
- $R^2_\text{adj}\le R^2$, with equality iff $n=0$ or $R^2=1$.
- Can be negative.
- Adding a feature increases $R^2_\text{adj}$ iff that feature's partial $F>1$, i.e., $|t|>1$ — a much weaker bar than statistical significance ($|t|>1.96$).
- Undefined at $m=n+1$; unstable when $n\to m$.
```

```{attention} Q&A
:class: dropdown
*When does it help?*
- Comparing nested linear models on the **same** in-sample data, cheaply, without a held-out set.

*When does it not?*
- Non-linear models, where "$n$" is not the effective degrees of freedom.
- Anything evaluated out-of-sample — held-out $R^2$ already penalizes useless features, so the adjustment is redundant and wrong.

*Better alternatives for model selection?*
- Cross-validated $R^2$/RMSE; or AIC/BIC, which penalize $n$ harder ($2$ and $\log m$ per parameter respectively) than the $|t|>1$ bar.
```

&nbsp;

## Multi-Label
### Subset Accuracy
- **What**: Fraction of samples whose predicted label **set** exactly matches the true set.
- **Why**: Per-label metrics can look excellent while not a single sample is entirely correct.
- **How**: Compare the two label sets per sample → require exact equality.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $L$: #labels.
    - $Y_i\subseteq\{1,\dots,L\}$: True label set of sample $i$.
    - $\hat{Y}_i$: Predicted label set of sample $i$.

Definition:

$$
\text{SA}=\frac{1}{m}\sum_{i=1}^{m}\mathbb{1}\left[\hat{Y}_i=Y_i\right]
$$

Properties:
- Also called exact match ratio.
- No partial credit: $L-1$ correct labels out of $L$ scores $0$.
- $\text{SA}\le1-\text{Hamming loss}$.
- Decays roughly geometrically in $L$ → near-$0$ for large label spaces.
```

```{attention} Q&A
:class: dropdown
*When is it the right metric?*
- The downstream action is all-or-nothing: an automated routing decision, a form that must be filled correctly, a structured output consumed by a parser.

*When is it misleading?*
- Large $L$ → everything scores near $0$ → no signal to compare models by.
- → Pair it with Hamming loss and micro/macro-F1.
```

&nbsp;

#### Jaccard Index
- **What**: Label-set intersection over union, averaged over samples.
- **Why**: Subset accuracy gives no partial credit, and Hamming loss is diluted by the many labels correctly predicted absent → neither grades a *partially* right label set.
- **How**: Intersect the predicted & true label sets → divide by their union → average.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $Y_i$: True label set of sample $i$.
    - $\hat{Y}_i$: Predicted label set of sample $i$.
    - $F_{1,i}$: F1 of sample $i$ over its own label set.

Definition:

$$
\text{J}=\frac{1}{m}\sum_{i=1}^{m}\frac{|\hat{Y}_i\cap Y_i|}{|\hat{Y}_i\cup Y_i|}
$$

Relation to F1, per sample:

$$
\text{J}_i=\frac{F_{1,i}}{2-F_{1,i}}
$$

Properties:
- Range $[0,1]$; $\text{J}_i=1$ iff the two sets match exactly → $\text{SA}\le\text{J}$.
- $\text{J}_i\le F_{1,i}$ always ← $2-F_{1,i}\ge1$. The gap peaks at $F_1=2-\sqrt{2}\approx0.59$.
- ❌TN at the label level, exactly like F1.
- $Y_i=\hat{Y}_i=\emptyset$ → $\frac{0}{0}$ → convention-dependent ($0$ or $1$) → state which.
```

```{attention} Q&A
:class: dropdown
*Jaccard vs F1?*
- Strictly monotone in each other **per sample** → identical ordering of two predictions on the same sample.
- Jaccard is always the harsher number: $F_1=0.5\Rightarrow\text{J}=\frac{1}{3}$.
- Averaged over samples or labels they CAN rank two models differently ← the mean of a nonlinear transform $\neq$ the transform of the mean.

*Which averaging?*
- Per sample → "how good is a typical predicted set".
- Per label with micro/macro → "how good is a typical label".
- Different questions, different numbers → say which one you ran.

*Where does the same quantity appear under another name?*
- Set similarity in dedup / entity matching, and IoU in detection & segmentation.
```

&nbsp;

### Hamming Loss
- **What**: Fraction of individual label slots predicted wrong.
- **Why**: Subset accuracy gives no partial credit → a near-perfect prediction is indistinguishable from a total miss.
- **How**: Count label-level mismatches → divide by $mL$.

```{note} Math
:class: dropdown
Definition:

$$
\text{HL}=\frac{1}{mL}\sum_{i=1}^{m}\sum_{l=1}^{L}\mathbb{1}\left[\hat{y}_{il}\neq y_{il}\right]
$$

Properties:
- $=1-$ label-wise accuracy → lower is better.
- Symmetric in FP and FN at the label level.
- Decomposable over (sample, label) pairs.
```

```{attention} Q&A
:class: dropdown
*Why is a low Hamming loss often meaningless?*
- Label sets are sparse: with $L=1000$ and $\approx3$ true labels per sample, predicting **nothing** gives $\text{HL}=0.003$.
- → Always report micro/macro-F1 over labels next to it.

*Hamming loss vs subset accuracy?*
- HL = per-label view, generous. SA = per-sample view, brutal.
- Both together bracket the truth: HL says how many slots are wrong, SA says how often *everything* was right.

*Does it capture label correlations?*
- ❌. It treats the $L$ labels as independent binary problems → a model that predicts a jointly impossible combination is not penalized.
```

&nbsp;

## Clustering
### Silhouette
- **What**: Per-point contrast between its own-cluster distance and its nearest-other-cluster distance.
- **Why**: ❌ground-truth labels → need an internal criterion; WCSS/inertia decreases monotonically with $k$ → useless for choosing $k$.
- **How**:
    1. $a$ = mean distance to the other points in its own cluster (cohesion).
    2. $b$ = mean distance to the points of the nearest other cluster (separation).
    3. Normalize the gap by $\max(a,b)$.
    4. Average over all points.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $c_i$: Cluster of point $i$.
    - $d(i,j)$: Distance between points $i$ and $j$.
    - $a(i)=\frac{1}{|c_i|-1}\sum_{j\in c_i,j\neq i}d(i,j)$: Mean intra-cluster distance.
    - $b(i)=\min_{c\neq c_i}\frac{1}{|c|}\sum_{j\in c}d(i,j)$: Mean distance to the nearest other cluster.

Definition:

$$
s(i)=\frac{b(i)-a(i)}{\max\left(a(i),b(i)\right)},\qquad S=\frac{1}{m}\sum_{i=1}^{m}s(i)
$$

Properties:
- $s(i)\in[-1,1]$; $s(i)=0$ by convention for a singleton cluster.
- $s(i)>0$ → closer to its own cluster than to any other; $s(i)<0$ → assigned to the wrong cluster.
- Requires $2\le k\le m-1$.
- $O(m^2)$ distance computations.
```

````{important} Code
:class: dropdown
```python
import numpy as np

def silhouette(X, labels):
    D = np.linalg.norm(X[:, None] - X[None, :], axis=-1)   ## pairwise distances
    s = np.zeros(len(X))
    for i in range(len(X)):
        own = labels == labels[i]
        if own.sum() == 1:                                 ## singleton -> 0 by convention
            continue
        a = D[i, own].sum() / (own.sum() - 1)              ## exclude the self-distance 0
        b = min(D[i, labels == c].mean() for c in set(labels) - {labels[i]})
        s[i] = (b - a) / max(a, b)
    return s.mean()

## Example: two tight, well-separated blobs
X = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 0.0], [5.1, 0.0]])
print(round(silhouette(X, np.array([0, 0, 1, 1])), 3))   ## 0.98
print(round(silhouette(X, np.array([0, 1, 0, 1])), 3))   ## -0.49  <- clusters split across blobs
```
````

```{dropdown} Table: Internal Indices
| Index | Direction | Built from |
|:--|:--|:--|
| Silhouette | Higher | Per-point cohesion vs separation |
| Davies-Bouldin | Lower | Mean over clusters of the worst within/between scatter ratio |
| Calinski-Harabasz | Higher | Between-cluster vs within-cluster variance, df-adjusted |
| WCSS / inertia | Lower | Within-cluster sum of squares |

- All three of the first row are label-free → usable for selecting $k$.
- WCSS alone cannot select $k$ ← strictly decreasing in $k$, hence the elbow heuristic.
- All of them reward compact, convex, roughly spherical clusters → all of them punish DBSCAN-style density clusters.
```

```{attention} Q&A
:class: dropdown
*How do you read the number?*
- $\approx1$ → dense, well-separated. $\approx0$ → clusters overlap / the point is on a boundary. $<0$ → probably mis-assigned.
- The per-point plot matters more than the mean: one clean cluster can carry a mediocre average.

*Cons?*
- $O(m^2)$ memory & time → subsample for large $m$.
- Assumes compact convex clusters → structurally biased toward K-Means solutions, and it will pick the wrong $k$ for elongated or nested shapes.
- Depends on the distance metric and on feature scaling → different preprocessing, different answer.
- Degrades in high dimensions ← distance concentration makes $a$ and $b$ converge.

*What is it actually used for?*
- Choosing $k$ ([K-Means](unsupervised.md#k-means)), and sanity-checking that any structure exists at all.

*Does a high silhouette mean the clustering is correct?*
- ❌. It means the partition is geometrically compact under the chosen metric. Ground-truth agreement needs ARI/NMI.
```

&nbsp;

### ARI
- **Name**: Adjusted Rand Index
- **What**: Chance-corrected agreement between two clusterings, counted over point pairs.
- **Why**: Cluster IDs are arbitrary permutations → accuracy is undefined; and the raw Rand Index is close to $1$ even for random partitions.
- **How**:
    1. Count point pairs the two partitions agree on (together in both, or apart in both).
    2. Subtract the count expected under random partitions with the same cluster sizes.
    3. Normalize by the maximum minus the expected.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $n_{ij}$: #points in cluster $i$ of partition $U$ and cluster $j$ of partition $V$.
    - $a_i=\sum_jn_{ij}$: Size of cluster $i$ in $U$.
    - $b_j=\sum_in_{ij}$: Size of cluster $j$ in $V$.

Rand Index:

$$
\text{RI}=\frac{\text{\#agreeing pairs}}{\binom{m}{2}}
$$

Adjusted Rand Index:

$$
\text{ARI}=\frac{\sum_{ij}\binom{n_{ij}}{2}-\frac{\sum_i\binom{a_i}{2}\sum_j\binom{b_j}{2}}{\binom{m}{2}}}{\frac{1}{2}\left[\sum_i\binom{a_i}{2}+\sum_j\binom{b_j}{2}\right]-\frac{\sum_i\binom{a_i}{2}\sum_j\binom{b_j}{2}}{\binom{m}{2}}}
$$

Properties:
- $\text{ARI}=1$ for identical partitions (up to relabeling), $\approx0$ for independent ones, negative when agreement is worse than chance.
- Permutation-invariant, and defined when $U$ and $V$ use different numbers of clusters — but NOT invariant to changing them: refining a partition drives ARI down.
- Symmetric in $U$ and $V$.
```

````{important} Code
:class: dropdown
```python
from math import comb
import numpy as np

def ari(u, v):
    C = np.zeros((u.max() + 1, v.max() + 1), int)
    np.add.at(C, (u, v), 1)                        ## contingency table of the two labelings
    ## count point PAIRS falling in the same cell / same row / same column
    same = sum(comb(x, 2) for x in C.ravel())
    a = sum(comb(x, 2) for x in C.sum(1))
    b = sum(comb(x, 2) for x in C.sum(0))
    exp = a * b / comb(C.sum(), 2)                 ## expectation with the marginals fixed
    denom = 0.5 * (a + b) - exp
    ## denom == 0 <=> both partitions are trivial (one cluster, or all singletons)
    return 1.0 if denom == 0 else (same - exp) / denom

## Example
u = np.array([0, 0, 1, 1])
print(round(ari(u, np.array([1, 1, 0, 0])), 3))   ## 1.0   <- same partition, renamed
print(round(ari(u, np.array([0, 1, 0, 1])), 3))   ## -0.5  <- agreement below chance
```
````

```{attention} Q&A
:class: dropdown
*Why is the raw Rand Index unusable?*
- Most pairs are "apart in both" simply because there are many clusters → RI drifts toward $1$ with $k$ and has no fixed baseline.
- ARI fixes the baseline at $0$ by subtracting the permutation-model expectation.

*ARI vs NMI?*
- ARI counts pairs → directly penalizes splitting one true cluster in two or merging two.
- NMI counts information → a pure refinement of the truth still scores high.
- ARI is chance-corrected; NMI is not (AMI is).

*When does ARI mislead?*
- Very unbalanced cluster sizes → pair counts are dominated by the largest cluster, so errors on small clusters are nearly free.

*Why not just match clusters to classes and compute accuracy?*
- That requires solving an assignment problem and breaks when the two partitions have different $k$; ARI needs neither.
```

&nbsp;

### NMI
- **Name**: Normalized Mutual Information
- **What**: Mutual information between two labelings, rescaled to $[0,1]$.
- **Why**: Raw MI is unbounded and grows with the number of clusters → two clusterings with different $k$ cannot be compared.
- **How**: Divide $I(U;V)$ by an average of the two entropies.

```{note} Math
:class: dropdown
Notations:
- Misc:
    - $U,V$: The two labelings, viewed as discrete random variables.
    - $H(\cdot)$: Entropy.
    - $I(U;V)=H(U)+H(V)-H(U,V)$: Mutual information.

Definition:

$$
\text{NMI}=\frac{I(U;V)}{\text{mean}\left(H(U),H(V)\right)}
$$

Adjusted variant:

$$
\text{AMI}=\frac{I(U;V)-\mathbb{E}[I(U;V)]}{\text{mean}\left(H(U),H(V)\right)-\mathbb{E}[I(U;V)]}
$$

Properties:
- $\text{NMI}\in[0,1]$; $0$ iff independent.
- $\text{NMI}=1\Leftrightarrow$ identical up to relabeling holds for the arithmetic & geometric normalizers ONLY — under $\min$, any refinement of the coarser labeling also scores $1$.
- The normalizer is a free choice — arithmetic, geometric, $\min$ or $\max$ mean — and each gives a different number → state which one.
- Permutation-invariant and symmetric in $U,V$.
- ❌chance-corrected → rises as $k$ grows, even for random labelings.
```

```{attention} Q&A
:class: dropdown
*Why does NMI reward too many clusters?*
- ❌chance-corrected: $\mathbb{E}[I]$ between two **random** labelings grows with $k$ → a finer random clustering scores higher for free.
- ❌ Not because splitting always helps — under the arithmetic normalizer, splitting a perfect clustering into singletons drops NMI from $1$ to $\frac{2}{3}$.
- → Never use NMI to select $k$. Use AMI, or fix $k$ and use it only to compare methods.

*NMI vs AMI?*
- AMI subtracts the expected MI under a random permutation model → baseline $0$, comparable across $k$.
- AMI costs more to compute and is the correct default whenever the two labelings have different granularity.

*Related decomposition?*
- Homogeneity (each cluster contains one class) and completeness (each class lands in one cluster); V-measure is their harmonic mean and equals NMI under the arithmetic normalizer.

*NMI vs ARI in practice?*
- Report both. Large unbalanced clusters → trust NMI more; refinement/merging errors → trust ARI more; they disagree exactly when one of those two failure modes is present.
```

&nbsp;
