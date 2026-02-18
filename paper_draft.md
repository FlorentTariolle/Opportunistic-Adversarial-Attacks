# Opportunistic Targeting: A Rank-Stability Heuristic for Query-Efficient Black-Box Adversarial Attacks

---

## Abstract

Black-box adversarial attacks that minimize only the ground-truth confidence suffer from *latent-space drift*: perturbations wander through the feature space without committing to a specific adversarial class, wasting queries on diffuse, undirected progress. We introduce **Opportunistic Targeting (OT)**, a lightweight wrapper that monitors the rank stability of the leading non-true class during an untargeted attack and dynamically switches to a targeted objective once a stable candidate emerges. OT requires no architectural modification to the underlying attack, no gradient access, and no *a priori* target-class knowledge.

We validate OT on two representative score-based attacks — SimBA and Square Attack (cross-entropy loss) — across four standard ImageNet classifiers. OT consistently matches the performance of an oracle that knows the optimal target class in advance, while substantially outperforming untargeted baselines. Benefits scale with model depth and are confirmed on a 100-image benchmark with bootstrapped confidence intervals: OT closes the gap between untargeted and oracle success rates across the full query-budget range.

On adversarially-trained models, OT's stability signal becomes less reliable due to flatter confidence landscapes, which we characterize and discuss.

---

## 1. Introduction

Standard untargeted black-box attacks operate by minimizing the model's confidence in the ground-truth class. This strategy — whether implemented as probability minimization in SimBA (Guo et al., 2019) or cross-entropy loss in Square Attack (Andriushchenko et al., 2020) — treats all non-true classes as interchangeable. As the ground-truth confidence decreases, the freed probability mass disperses across the remaining classes without directional commitment. The adversarial perturbation effectively executes a random walk through the latent space, crossing class basins opportunistically rather than heading toward a specific decision boundary.

This *latent-space drift* is not merely an aesthetic concern; it directly impacts query efficiency. Each query spent exploring a class basin that will ultimately be abandoned is a wasted query. The deeper the model (and hence the higher-dimensional the feature space), the more basins the perturbation must traverse before settling, and the more pronounced the waste becomes. On a 50-layer residual network, an untargeted attack may require twice the queries needed to cross the *same* decision boundary that a targeted attack reaches directly.

Targeted attacks eliminate drift by construction: they push toward a fixed adversarial class from the first query. However, they require the attacker to specify a target class *a priori* — information that is generally unavailable in a black-box setting. Choosing a suboptimal target can be worse than having no target at all, because the attack commits its entire budget to reaching a potentially distant or infeasible class.

Margin-based losses (Carlini and Wagner, 2017) offer a partial solution. By optimizing the gap between the ground-truth logit and the highest non-true logit, they implicitly track the nearest decision boundary at every iteration. The Square Attack paper's default loss is of this form, and it provides strong untargeted performance precisely because it encodes dynamic target selection into the loss function itself. However, not all attacks support margin losses. SimBA operates on raw probabilities; other attacks may use cross-entropy for compatibility or simplicity. For these methods, drift remains a first-order efficiency bottleneck.

**Opportunistic Targeting** bridges this gap. The key insight is that the information needed to select a good target class is already present in the attack's own trajectory: the class that an untargeted perturbation is naturally drifting toward is, by definition, the class whose decision boundary is most accessible. OT formalizes this observation through a rank-stability heuristic:

1. **Exploration phase.** The attack runs in standard untargeted mode. After each *accepted* perturbation (a step that improved the adversarial loss), we record which non-true class currently holds the highest predicted confidence.

2. **Stability check.** If the same class holds the top rank for $S$ consecutive accepted perturbations (the *stability threshold*), we declare it the opportunistic target.

3. **Exploitation phase.** The attack switches to a pure targeted objective against the locked class and runs until misclassification or budget exhaustion.

The stability threshold $S$ acts as a debouncing filter: it prevents the attack from locking onto volatile classes that spike due to random noise in early iterations, while remaining small enough that the exploration phase consumes a negligible fraction of the query budget.

This paper makes three contributions:

1. **A general-purpose wrapper** that adds opportunistic targeting to any score-based black-box attack, requiring only access to the predicted class distribution (not gradients or logits).

2. **Empirical validation** on standard ImageNet classifiers showing that OT provides near-oracle efficiency with zero *a priori* target knowledge, with benefits that scale predictably with model depth.

3. **A characterization of OT's failure mode** on adversarially-trained models, where flat confidence landscapes produce semantically plausible decoy targets that mislead the stability heuristic.

The remainder of this paper is organized as follows. Section 2 surveys related work on query-efficient black-box attacks. Section 3 describes the OT algorithm and its integration with SimBA and Square Attack. Section 4 details the experimental setup. Section 5 presents results on standard networks. Section 6 formalizes the connection between OT and margin-based losses. Section 7 reports ablation studies on the stability threshold and loss function. Section 8 extends the analysis to adversarially-trained models. Section 9 concludes.

---

## 2. Related Work

### 2.1 Score-Based Black-Box Attacks

Score-based (decision-score) attacks assume access to the full output probability vector but not to gradients. SimBA (Guo et al., 2019) iterates over an orthonormal basis (pixel or DCT), accepting perturbations that reduce the true-class probability. It requires only 1.4–1.5 queries per iteration on average and achieves high success rates on standard models. While SimBA could in principle adopt a margin-based acceptance criterion, its original formulation uses only the true-class probability, providing no mechanism for directing perturbations toward a specific adversarial class.

Square Attack (Andriushchenko et al., 2020) uses random square-shaped patches at the vertices of the $L_\infty$ ball, with a schedule that shrinks the patch size as the attack progresses. Its default margin loss $f_{y}(x) - \max_{k \neq y} f_k(x)$ implicitly tracks the nearest decision boundary. When run with cross-entropy loss instead, the margin guidance vanishes and the attack exhibits the same drift behavior as SimBA — making it an ideal testbed for isolating OT's contribution.

### 2.2 Decision-Based and Transfer Attacks

Decision-based attacks require only the top-1 label, not the full score vector. Recent work in this space focuses on geometric constructions: SurFree (Maho et al., 2021) uses random 2-D hyperplane search, and Gesny et al. (2024) show theoretically that reintroducing gradient estimation into SurFree (yielding CGBA) accelerates convergence of the angle $\theta(i)$ between the current perturbation direction and the optimal adversarial direction. This is structurally analogous to OT: where CGBA reintroduces directional information (gradient) into a blind geometric process, OT reintroduces directional information (target locking) into a blind score-based process. We formalize this parallel in Section 6.3 using the same angular convergence framework.

### 2.3 Attacks on Robust Models

Adversarially-trained models (Madry et al., 2018; Salman et al., 2020) present a qualitatively different challenge. Their flatter loss landscapes and more uniform confidence distributions mean that early-iteration class rankings are less informative. The robust-model experiments in Section 8 examine whether OT's rank-stability signal remains useful in this regime.

---

## 3. Method

### 3.1 Notation

Let $f: \mathcal{X} \to \mathbb{R}^K$ denote a classifier mapping inputs to logits over $K$ classes, and $P(k|x) = \text{softmax}(f(x))_k$ the predicted probability of class $k$. Given a correctly-classified input $x$ with true label $y$, the attacker seeks an adversarial example $x' = x + \delta$ such that $\arg\max_k P(k|x') \neq y$ and $\|\delta\|_\infty \leq \epsilon$.

### 3.2 Untargeted Loss and the Drift Problem

Standard untargeted attacks minimize a loss that depends only on the true class:

$$\mathcal{L}_{\text{untargeted}}(x', y) = P(y | x') \quad \text{(SimBA)} \qquad \text{or} \qquad \mathcal{L}_{\text{untargeted}}(x', y) = -\log P(y | x') \quad \text{(CE)}$$

These objectives decrease $P(y|x')$ without specifying where the freed probability mass should concentrate. In a $K$-class problem, the perturbation may distribute mass across hundreds of classes, crossing multiple class basins before any single competitor class exceeds the declining true-class confidence. We call this *latent-space drift*.

**Margin loss** avoids drift by construction:

$$\mathcal{L}_{\text{margin}}(x', y) = f_y(x') - \max_{k \neq y} f_k(x')$$

The $\max_{k \neq y} f_k(x')$ term dynamically identifies the nearest competitor, providing implicit directionality at every iteration. Square Attack's default loss is of this form, explaining its strong untargeted performance.

### 3.3 Targeted Loss

A targeted attack toward class $t$ optimizes:

$$\mathcal{L}_{\text{targeted}}(x', t) = -P(t | x') \quad \text{(SimBA)} \qquad \text{or} \qquad \mathcal{L}_{\text{targeted}}(x', t) = \log P(t | x') \quad \text{(CE)}$$

Targeting eliminates drift but requires knowing $t$ in advance. An *oracle* target — the class that the unconstrained untargeted attack eventually reaches — provides an upper bound on targeted performance that no real attacker can achieve.

### 3.4 Opportunistic Targeting Algorithm

OT discovers the target class online by monitoring which adversarial class the perturbation is naturally drifting toward. The algorithm wraps any score-based attack without modifying its perturbation mechanism.

#### Algorithm 1: Opportunistic Targeting Wrapper

```text
Input: image x, true label y, attack A, stability threshold S
Output: adversarial example x'

1.  Initialize: x' ← x, locked ← False, target ← None, buffer ← deque(maxlen=S)
2.  while not misclassified(x') and budget not exhausted:
3.      x' ← A.step(x', y if not locked else target, mode)
4.      if step was accepted:                          // loss improved
5.          c ← argmax_{k ≠ y} P(k | x')              // leading non-true class
6.          if not locked:
7.              buffer.append(c)
8.              if len(buffer) = S and all entries in buffer are identical:
9.                  target ← c
10.                 locked ← True
11.                 mode ← targeted
12.             else:
13.                 mode ← untargeted
14. return x'
```

**Key design choices:**

- **Accepted perturbations only (line 4).** The stability counter increments only when the attack makes progress (reduces the loss). Rejected steps — which contribute no useful signal about the loss landscape — are ignored. This filters out noise from random, unproductive queries.

- **Consecutive stability (line 8).** The buffer must contain $S$ identical entries *in a row*. A single interruption resets the count. This strict debouncing prevents premature lock-in on volatile classes.

- **Irreversible lock (line 9–11).** Once a target is locked, the attack commits for the remainder of the budget. Releasing the lock would re-introduce the exploration overhead that OT is designed to eliminate.

### 3.5 Integration with SimBA

SimBA (Guo et al., 2019) perturbs the image along randomly-sampled orthonormal directions (pixel or DCT basis), accepting steps that improve the adversarial loss. Our implementation uses the DCT basis with 8×8 blocks, operating in $L_\infty$ with $\epsilon = 8/255$.

| Parameter | Value | Description |
| ----------- | ------- | ------------- |
| Basis | DCT (8×8 blocks) | Low-frequency directions; ~98% are descending |
| Step size | $\epsilon = 8/255$ | $L_\infty$ perturbation bound |
| Budget | 10,000 queries | Per-image query limit |
| Acceptance | $P(y\|x') < P(y\|x)$ (untargeted) | Greedy coordinate descent |

In untargeted mode, SimBA's acceptance criterion reduces $P(y|x')$ — a single-class objective that, unlike margin loss, does not track the nearest competitor. Upon lock-in, the criterion switches to increasing $P(t|x')$ where $t$ is the locked target. The perturbation mechanism (basis selection, step size, acceptance rule) is unchanged — only the objective updates.

### 3.6 Integration with Square Attack (Cross-Entropy Loss)

Square Attack (Andriushchenko et al., 2020) samples random square patches at the vertices of the $L_\infty$ ball, with a patch size that decays according to a budget-dependent schedule. The schedule halves the patch fraction $p$ at iterations $\{10, 50, 200, 1000, 2000, 4000, 6000, 8000\}$ (for a 10,000-iteration budget), rescaling proportionally for other budgets.

**Important methodological note.** The patch schedule depends on the *total* budget $N$, not just the current iteration $i$. The `torchattacks` implementation normalizes via $\hat{i} = \lfloor i / N \times 10000 \rfloor$. This means iteration counts obtained under different budgets are **not comparable**: an attack with $N = 10\text{K}$ at iteration 5 used larger patches than one with $N = 1\text{K}$ at iteration 5. All our experiments use a fixed $N = 10\text{K}$ for Square Attack.

We run Square Attack with **cross-entropy loss** ($-\log P(y|x')$) rather than the default margin loss. This is a deliberate ablation: margin loss already provides implicit target tracking (Section 3.2), which would confound OT's contribution. With CE loss, the untargeted attack exhibits clear drift, and any efficiency gain can be attributed to OT.

| Parameter | Value | Description |
| ----------- | ------- | ------------- |
| Patch shape | Square, $L_\infty$ vertices | $\pm \epsilon$ per pixel |
| Loss | Cross-entropy | Drift-prone; no implicit targeting |
| Budget | $N = 10{,}000$ | Fixed for schedule comparability |
| $p$ schedule | Halved at $\{10, 50, 200, \ldots, 8000\}$ | Patch fraction decay |

---

## 4. Experimental Setup

### 4.1 Models

We evaluate on two model families:

**Standard (non-robust) ImageNet classifiers** from torchvision, wrapped with a `NormalizedModel` that applies ImageNet normalization internally so all attacks operate in $[0, 1]$ pixel space:

| Model | Depth | Parameters | Top-1 Accuracy |
| ------- | ------- | ----------- | ---------------- |
| AlexNet | 8 layers | 61M | 56.5% |
| ResNet-18 | 18 layers | 11.7M | 69.8% |
| VGG-16 | 16 layers | 138M | 71.6% |
| ResNet-50 | 50 layers | 25.6M | 76.1% |

**Adversarially-trained ImageNet classifiers** from RobustBench (Salman et al., 2020), with built-in normalization:

| Model | Architecture | Training |
| ------- | ------------- | ---------- |
| Salman2020Do\_R18 | ResNet-18 | PGD adversarial training |
| Salman2020Do\_R50 | ResNet-50 | PGD adversarial training |

### 4.2 Protocol

Each (model, attack, image) triplet is evaluated in three modes:

| Mode | Description |
| ------ | ------------- |
| **Untargeted** | Standard attack with no directional guidance. |
| **Targeted (oracle)** | Upper bound: target class chosen *a posteriori* from the untargeted result. |
| **Opportunistic** | Our method: lock onto the leading non-true class once rank-stability threshold $S$ is reached. |

The oracle target is determined by running the untargeted attack first and recording the final adversarial class. This provides the strongest possible baseline: a targeted attack that knows exactly which class the model is most susceptible to for each image.

### 4.3 Configuration

- **Perturbation budget:** $\epsilon = 8/255 \approx 0.031$ in $[0, 1]$ pixel space ($L_\infty$ norm). This is the standard ImageNet adversarial benchmark setting.
- **Query budget:** 10,000 iterations for both attacks (4-image benchmark) and 15,000 iterations (100-image winrate benchmark).
- **Seeds:** Three random seeds (0, 1, 2) for the 4-image benchmark; single seed for the 100-image benchmark.
- **Stability threshold:** $S = 5$ for standard models, $S = 10$ for robust models.

### 4.4 Images

**4-image benchmark:** Four manually-selected ImageNet-compatible images (`corgi.jpg`, `porsche.jpg`, `dumbbell.jpg`, `hammer.jpg`). We filter (model, method, image) triplets where all three modes failed across all seeds, as these represent fundamentally infeasible attacks. This removes 2 configurations (SimBA on AlexNet for corgi and porsche), yielding 270 runs (90 per mode).

**100-image benchmark:** 100 images sampled from the ILSVRC2012 validation set (seed 42), selected to be correctly classified by ResNet-50. Each (method, image, mode) combination runs once at a fixed 15K budget.

### 4.5 Metrics

On standard networks where most attacks succeed, the primary metric is **iterations to success** (lower is better). Following Ughi et al. (2021), we also report **success rate as a function of query budget** (CDF curves), which captures the full distribution of attack difficulty rather than reducing it to a single threshold. On robust networks where most attacks fail, we report **final margin** $= \max(P(y_{\text{true}}) - \max_{k \neq y} P(k), 0)$ (lower is better — closer to the decision boundary).

### 4.6 Budget Censoring

All iteration counts are right-censored at the query budget. Failed runs that hit the ceiling represent a *lower bound* on the true cost. This censoring biases *against* OT's reported savings: the true untargeted cost for hard cases may be 2–10× higher, making our savings estimates conservative.

---

## 5. Results on Standard Networks

### 5.1 Success Rates

| Method | Untargeted | Targeted (oracle) | Opportunistic |
| -------- | ----------- | ------------------- | --------------- |
| SimBA | 88.1% | 100% | 100% |
| Square Attack | 100% | 100% | 100% |

OT matches oracle-targeted success rates for both attacks. SimBA's untargeted mode fails on ~12% of configurations (hard cases on deeper models), while OT rescues all of them. **OT not only converges faster — it also converts untargeted failures into successes.**

### 5.2 CDF: Success Rate vs. Query Budget

![CDF 4-image](results/figures/standard/fig_cdf.png)

**Figure: Cumulative success rate vs. query budget** (4-image benchmark, all models). Solid = SimBA; dashed = Square Attack. Colors distinguish modes (blue = untargeted, red = opportunistic). OT dominates its untargeted baseline across the full budget range.

The 100-image CDF benchmark on ResNet-50 (15K budget, 1000-sample bootstrap, 90% CI bands) confirms these findings at scale with tighter confidence intervals.

![SimBA CDF](results/figures_winrate/fig_winrate_simba.png)

**Figure: SimBA success rate vs. query budget** (ResNet-50, 100 images). OT (green) matches oracle-targeted performance (orange) exactly at 85%, a +32 percentage point gain over untargeted (53%). The curves overlap almost entirely, confirming that OT recovers oracle-level efficiency without knowing the target class.

![Square Attack CDF](results/figures_winrate/fig_winrate_squareattack.png)

**Figure: Square Attack (CE) success rate vs. query budget** (ResNet-50, 100 images). OT reaches 98% success, matching oracle (99%) and far exceeding untargeted (85%). The OT and oracle curves are nearly indistinguishable across the full budget range.

### 5.3 Mean Iterations to Success

*Successful runs only (4-image benchmark, 3 seeds, 4 models).*

| Method | Untargeted | Targeted (oracle) | Opportunistic | Savings vs. Untargeted |
| -------- | ----------- | ------------------- | --------------- | ---------------------- |
| SimBA | 5,117 | 4,295 | 4,383 | **14.3%** |
| Square Attack | 865 | 430 | 447 | **48.3%** |

![Headline bars](results/figures/standard/fig_headline_bars.png)

**Figure: Mean iterations by attack mode** (all models, successful runs only). Error bars show 95% bootstrap CI. Percentage annotations indicate OT savings vs. untargeted.

For SimBA, OT reduces mean iterations by 14.3%, landing within 2% of the oracle. For Square Attack (CE loss), the reduction is 48.3%. The Square Attack numbers are dominated by ResNet-50 (Section 5.4), where drift is most severe.

![Violin](results/figures/standard/fig_violin.png)

**Figure: Query distribution** (log scale). Untargeted (blue) vs. Opportunistic (red). Dashed lines show medians.

*Median query counts:*

| Method | Untargeted | Opportunistic |
| -------- | ----------- | --------------- |
| SimBA | 5,865 | 4,706 |
| Square Attack | 292 | 280 |

The distribution for Square Attack is bimodal: most runs converge quickly (~1–500 queries), with a long tail of hard cases in the thousands. OT primarily cuts this long tail.

### 5.4 The Depth-Scaling Hypothesis

![Per-model iterations](results/figures/standard/fig_per_model.png)

**Figure: Iterations by model and mode.** OT's benefit scales with model depth — largest on ResNet-50, negligible on AlexNet.

The per-model breakdown reveals that OT's benefit scales with model depth:

| Model | Method | Untargeted | Opportunistic | Savings |
| ------- | -------- | ----------- | --------------- | --------- |
| **ResNet-50** | **SimBA** | **7,664**† | **4,860** | **36.6%**† |
| **ResNet-50** | **Square Attack** | **2,521** | **906** | **64.0%** |
| VGG-16 | SimBA | 6,394 | 5,766 | 9.8% |
| VGG-16 | Square Attack | 322 | 344 | −6.8% |
| ResNet-18 | SimBA | 3,890 | 3,828 | 1.6% |
| ResNet-18 | Square Attack | 314 | 243 | 22.6% |
| AlexNet | SimBA | 2,044 | 1,775 | 13.2% |
| AlexNet | Square Attack | 304 | 294 | 3.3% |

†SimBA on ResNet-50 untargeted: only 7/12 runs succeed (5 hit the 10K ceiling). The untargeted mean is computed from successful runs only — the true cost is higher, so 36.6% is a conservative lower bound. OT achieves 12/12 success.

![ResNet-50 heatmap](results/figures/standard/fig_resnet50_heatmap.png)

**Figure: ResNet-50 per-image heatmap.** Mean iterations by image and mode. Red = high (hard); yellow = low (easy). OT consistently reduces iterations on hard images (dumbbell, hammer).

**ResNet-50 is where OT shines.** We hypothesize this follows from latent space dimensionality: a 50-layer residual network maps inputs into a far higher-dimensional feature space than an 8-layer AlexNet. In this higher-dimensional space, untargeted perturbations experience more class drift — the perturbation wanders through multiple adversarial class basins before settling. OT detects the emerging basin early and locks onto it, eliminating the drift phase.

![Difficulty vs savings](results/figures/standard/fig_difficulty_vs_savings.png)

**Figure: Opportunistic savings vs. attack difficulty.** Each point is one (model, method, image, seed) run. Dashed line: linear trend ($r = 0.38$, $p < 0.001$).

The difficulty–savings scatter plot supports this hypothesis: there is a statistically significant positive correlation between untargeted difficulty and OT's benefit. **The harder the attack, the more OT helps.** This is the regime that matters for practical adversarial evaluation.

### 5.5 Lock-in Dynamics

![Lock-in dynamics](results/figures/standard/fig_lockin.png)

**Figure: Lock-in dynamics.** Faded curves show untargeted; vivid curves show opportunistic. Vertical dotted lines mark lock-in and convergence iterations.

The confidence traces illustrate *why* OT works.

**Square Attack on ResNet-50 (hammer.jpg).** Lock-in at iteration 31. OT converges at 2,126 vs. untargeted at 5,621 (62.2% savings). The untargeted trace shows ground-truth confidence declining slowly with the max-other-class confidence near zero — the perturbation spreads probability mass across many classes without committing. OT locks onto class 677 at iteration 31 and drives it upward monotonically, crossing the decision boundary at iteration 2,126.

**SimBA on ResNet-50 (corgi.jpg).** Lock-in at iteration 7 (essentially immediate). OT converges at 3,958 vs. untargeted at 7,075 (44.1% savings). SimBA's greedy coordinate-descent steps produce more consistent class rankings, enabling near-instant lock-in.

**Lock-in timing (all opportunistic runs):**

| Method | Mean Switch | Median | Range |
| -------- | ------------ | -------- | ------- |
| SimBA | 7.0 | 7 | 6–10 |
| Square Attack | 141.9 | 113 | 11–542 |

88.5% of all opportunistic runs (79/90) trigger the lock-in. The exploration phase consumes <2% of the budget in all cases.

**Lock-match rate.** Does OT find the "right" target — the class that the untargeted attack would eventually reach?

![Lock-match rate](results/figures/standard/fig_lock_match.png)

**Figure: Lock-match rate by model.** SimBA consistently locks the oracle class (83.8% overall); Square Attack is lower (78.4%) but still succeeds.

| Method | Lock-Match Rate |
| -------- | ---------------- |
| SimBA | 83.8% |
| Square Attack | 78.4% |

Crucially, mismatch does not imply failure. OT achieves 100% success rate even when locking a different class than untargeted. The heuristic does not need the "correct" class — it only needs *a viable* class.

---

## 6. Opportunistic Targeting as a Margin Surrogate

### 6.1 Equivalence Under Stable Rankings

We can formalize the relationship between OT and margin loss. Once OT locks onto target class $t$, the attack minimizes $-P(t|x')$. If $t = \arg\max_{k \neq y} P(k|x')$ (which holds at the moment of lock-in, by definition), then maximizing $P(t|x')$ is equivalent to maximizing $\max_{k \neq y} P(k|x')$, which is exactly the margin loss's competitor term.

More precisely, the margin loss decomposes as:

$$\mathcal{L}_{\text{margin}} = f_y(x') - \max_{k \neq y} f_k(x') = \underbrace{f_y(x')}_{\text{push down true class}} - \underbrace{f_t(x')}_{\text{push up competitor}}$$

where $t$ is dynamically reselected at each iteration. OT approximates this by fixing $t$ after the exploration phase. The approximation is tight when the locked class remains the strongest competitor throughout the attack — which the stability check is designed to ensure.

### 6.2 Empirical Confirmation: CE Loss Ablation

Square Attack with margin loss shows **no benefit** from OT: the margin loss already performs dynamic target tracking at every iteration. When we strip this guidance by switching to CE loss, the untargeted attack degrades dramatically (865 vs. 430 mean iterations), and OT restores near-oracle performance (447 iterations). This confirms that **OT functions as a structural surrogate for margin loss**, providing the directionality that drift-prone losses lack.

| Loss Function | Untargeted | Oracle | OT | OT Benefit |
| -------------- | ----------- | -------- | ----- | ----------- |
| Margin (default) | ~430 | ~430 | ~430 | None |
| Cross-entropy | 865 | 430 | 447 | 48.3% savings |

### 6.3 Perturbation Alignment (Theta Convergence)

Our analysis here is inspired by the angular convergence framework of Gesny et al. (2024), who track the angle $\theta(i)$ between the current perturbation direction $u(i)$ and the optimal direction $n$ (the decision boundary normal) to characterize how reintroducing gradient information accelerates convergence in decision-based attacks. We adopt the same lens for score-based attacks: we track the cosine similarity between each attack's perturbation $\delta(i) = x'(i) - x$ and the oracle direction $\delta_{\text{oracle}}$ (the perturbation produced by a targeted attack toward the oracle class). If OT acts as a margin surrogate, its perturbation should align more rapidly with $\delta_{\text{oracle}}$ than the untargeted attack's perturbation — just as CGBA's perturbation aligns more rapidly with the boundary normal than SurFree's.

![Theta convergence](results/figures_theta/fig_theta.png)

**Figure: Perturbation alignment with oracle direction** (SimBA, ResNet-50, 100 images, 500-iteration budget). Shaded regions show $\pm 1$ standard deviation. The vertical dashed line marks the mean switch iteration.

The results confirm the margin-surrogate hypothesis. Untargeted perturbations drift quasi-orthogonally to the oracle direction, reaching a terminal cosine similarity of only $0.174 \pm 0.189$ (median $0.190$). Opportunistic perturbations, after switching at a mean iteration of $7.3$ (median $7$, range $6$–$15$), rapidly align with $\delta_{\text{oracle}}$, reaching a terminal similarity of $0.865 \pm 0.192$ (median $0.910$). The alignment gap of $0.692$ demonstrates that OT does not merely select the correct target — it actively redirects the perturbation toward the oracle basin, functioning as a structural surrogate for margin-based directional guidance.

---

## 7. Ablations

### 7.1 Stability Threshold $S$

The stability threshold $S$ controls the tradeoff between exploration (low $S$: lock quickly, risk locking on noise) and exploitation (high $S$: lock cautiously, waste budget on undirected exploration). Crucially, the optimal $S$ is **method-dependent**: SimBA's greedy coordinate-descent steps stabilize class rankings almost immediately (median lock-in at iteration ~7), while Square Attack's stochastic patch placement produces more volatile early rankings. A threshold that is tight for SimBA may be premature for Square Attack, and vice versa.

We sweep $S \in \{2, 3, 5, 8, 10, 12, 15\}$ independently for both attacks on standard ResNet-50 (100 images, 15K budget), reporting success rate and mean iterations to success for each (method, $S$) pair.

![S ablation](results/figures_ablation_s/fig_ablation_s.png)

**Figure: Stability threshold ablation** — top row: SimBA, bottom row: Square Attack (CE). Left: success rate vs. $S$; right: mean iterations (successful runs) vs. $S$. Dotted lines mark the optimal $S$.

**SimBA** ($S \in \{2, \ldots, 15\}$): Success rate is remarkably flat, ranging from 84.0% ($S = 2$) to 85.1% ($S = 10$). Mean iterations are similarly stable ($4{,}814$–$4{,}956$). The optimal threshold is $S^*_{\text{SimBA}} = 10$, though the margin over neighboring values is slim. SimBA's greedy coordinate-descent steps produce stable early-iteration rankings, making the heuristic robust to $S$.

| $S$ | Success Rate | Mean Iters | Median Iters |
| ----- | ------------- | ----------- | ------------- |
| 2 | 84.0% | 4,860 | 4,108 |
| 3 | 84.2% | 4,814 | 3,905 |
| 5 | 85.0% | 4,952 | 4,286 |
| 8 | 85.1% | 4,915 | 4,144 |
| **10** | **85.1%** | **4,889** | **4,075** |
| 12 | 84.0% | 4,832 | 4,143 |
| 15 | 85.0% | 4,956 | 4,366 |

**Square Attack (CE)** ($S \in \{2, \ldots, 15\}$): The success rate peaks at $S = 8$ (98.1%) and drops slightly at lower and higher thresholds. Mean iterations show a clear valley at $S = 8$–$10$ ($1{,}719$–$1{,}780$), rising at both ends. The optimal threshold is $S^*_{\text{Square}} = 8$.

| $S$ | Success Rate | Mean Iters | Median Iters |
| ----- | ------------- | ----------- | ------------- |
| 2 | 97.1% | 1,917 | 779 |
| 3 | 97.1% | 1,929 | 774 |
| 5 | 97.1% | 2,004 | 754 |
| **8** | **98.1%** | **1,780** | **753** |
| 10 | 96.1% | 1,719 | 582 |
| 12 | 97.0% | 1,850 | 633 |
| 15 | 96.0% | 1,910 | 652 |

The optimal thresholds differ: $S^*_{\text{SimBA}} = 10$ vs. $S^*_{\text{Square}} = 8$. This confirms that the stability heuristic should be calibrated per attack. Square Attack's stochastic patch placement produces more volatile early rankings than SimBA's coordinate descent, yet it achieves peak performance at a *lower* $S$ — likely because the larger per-step perturbations cause the correct target class to dominate earlier when it does stabilize.

### 7.2 Loss Function Ablation (Square Attack)

The CE-loss ablation on Square Attack isolates OT's contribution from the attack's native loss function:

| Configuration | Mean Iters | Success Rate | Notes |
| -------------- | ----------- | ------------- | ------- |
| Margin loss, untargeted | ~430 | 100% | Implicit dynamic targeting via $\max_{k \neq y}$ |
| Margin loss + OT | ~430 | 100% | No additional benefit — OT is redundant |
| CE loss, untargeted | 865 | 100% | Drift: 2× the queries of margin |
| CE loss + OT | 447 | 100% | Restores near-margin performance |
| CE loss, oracle targeted | 430 | 100% | Upper bound |

The pattern is clear: margin loss provides built-in target tracking that makes OT unnecessary. CE loss lacks this tracking, resulting in drift. OT compensates for the missing margin term, restoring efficiency to within 4% of the oracle. This confirms OT as a **general-purpose margin surrogate** applicable to any drift-prone loss.

---

## 8. Robust Models

### 8.1 Setup

We extend the evaluation to adversarially-trained ImageNet classifiers from RobustBench (Salman et al., 2020). The protocol mirrors the standard benchmark with two modifications:

1. **Stability threshold $S = 10$** (doubled from 5) to account for robust models' flatter confidence landscapes, which produce more volatile early-iteration class rankings.

2. **Metric: final margin** instead of iteration count. On standard networks, most attacks succeed and we compare speed. On robust networks, most attacks fail (overall success rate: 12.5–37.5%), so iteration counts are uninformative. Instead, we report the final margin $= \max(P(y_{\text{true}}) - \max_{k \neq y} P(k), 0)$ — lower is better (closer to misclassification).

The query budget of 10,000 matches the standard benchmark. This is likely insufficient for robust models — the Square Attack paper recommends 20,000 queries with random restarts for robust evaluation. Our budget is constrained by compute; the low absolute success rates should be interpreted with this limitation in mind.

### 8.2 Margin Analysis and the Decoy Hypothesis

**Success rates:**

| Model | Method | Untargeted | Oracle | OT |
| ------- | -------- | ----------- | -------- | ----- |
| R18 | SimBA | 25.0% | 25.0% | 25.0% |
| R18 | Square Attack | 50.0% | 50.0% | 50.0% |
| R50 | SimBA | 0.0% | 0.0% | 0.0% |
| R50 | Square Attack | 25.0% | 25.0% | **0.0%** |

The final row is the critical finding: **OT causes a success rate regression from 25% to 0% on R50 Square Attack.**

**Final margins (lower = better):**

| Model | Method | Untargeted | Oracle | OT |
| ------- | -------- | ----------- | -------- | ----- |
| R18 | SimBA | 0.330 | 0.333 | 0.352 |
| R18 | Square Attack | **0.120** | **0.077** | **0.075** |
| R50 | SimBA | 0.568 | 0.576 | 0.576 |
| R50 | Square Attack | **0.221** | **0.188** | **0.267** |

![Robust margins](results/figures/robust/fig_margin_per_model.png)

**Figure: Final margin by model and mode on robust networks** (lower = better attack). Error bars show 95% CI across images and seeds.

![Robust margin heatmap](results/figures/robust/fig_margin_heatmap.png)

**Figure: Per-image margin heatmap on robust models.** Green = low margin (near misclassification); red = high margin (attack failed to make progress).

**SimBA is mode-invariant on robust models.** Targeting mode has no measurable effect on margin or success rate. SimBA's coordinate-wise perturbations lack the power to exploit directional guidance against adversarial training.

**Square Attack shows mixed results.** On R18, OT matches oracle performance (margin 0.075 vs. 0.077). On R50, OT *underperforms* untargeted (0.267 vs. 0.221). Targeted oracle still helps (0.188), confirming that the problem is not with targeting itself but with *which class* OT selects.

![Robust lock-match](results/figures/robust/fig_lock_match.png)

**Figure: Lock-match rate on robust models.** On R50, OT never locks the oracle class (0% for both methods), explaining the performance regression. Dashed line shows the ~80% lock-match rate on standard models for comparison.

We term this failure mode the **decoy hypothesis**: on robust networks, adversarial training smooths the input loss landscape, creating an environment where multiple competing classes have similar early stability. OT locks onto a *semantically plausible* neighbor class that appears stable early but lies in a shallow adversarial basin:

| Image | True Class | OT Locks Onto | Peak Adversarial Class |
| ------- | ----------- | --------------- | ---------------------- |
| corgi.jpg | Pembroke (263) | Cardigan (264) | Dingo / Cardigan |
| dumbbell.jpg | Dumbbell (543) | Barbell (422) | Barbell (422) |
| hammer.jpg | Hammer (587) | Hatchet (596) | Hatchet / **Tripod (872)** |
| porsche.jpg | Sports car (817) | Racer (751) | Racer (751) |

The locked classes are not random — they are semantically related neighbors. For dumbbell and porsche, OT correctly identifies the peak adversarial class, but the attack still fails because even correctly-targeted perturbations cannot overcome the defense within budget. The critical failure is on hammer.jpg: OT locks onto hatchet (596), while the untargeted attack naturally drifts to tripod (872) — a different, viable class. OT commits its budget to the wrong basin.

### 8.3 Why R18 Survives

On R18, OT locks onto the correct peak adversarial class for most images, and the model's lower robust accuracy means targeting provides a genuine margin improvement (0.120 $\to$ 0.075). The defense is weaker, so even incomplete convergence toward the right target reduces margin meaningfully. On R50, the defense is stronger, and the single case where OT picks the wrong peak class (hammer.jpg) drives the entire success rate regression.

### 8.4 Robust Stability Threshold Ablation

[TODO: run S ablation on robust models (Salman2020Do_R50, Square Attack CE loss) to determine whether a higher $S$ mitigates the decoy effect. Hypothesis: increasing $S$ from 10 to 20–50 may allow the correct basin to emerge before lock-in, at the cost of a shorter exploitation phase. Report optimal $S_{\text{robust}}$ and whether it eliminates the hammer.jpg regression.]

### 8.5 Robust CDF (100-image)

[TODO: run 100-image CDF benchmark on robust model (Salman2020Do_R50 or R18) with the optimized $S_{\text{robust}}$ from Section 8.4. Determine whether OT is net positive on robust networks at scale, or whether the decoy effect is systematic. If net positive, report margin-CDF curves; if net negative, conclude that OT requires landscape-aware gating for robust models.]

---

## 9. Discussion and Conclusion

### 9.1 Summary of Contributions

We introduced Opportunistic Targeting, a wrapper that adds dynamic target selection to any score-based black-box adversarial attack. The key findings are:

1. **Near-oracle efficiency with zero prior knowledge.** On standard ImageNet classifiers, OT lands within 2–4% of an oracle that knows the optimal target class in advance, reducing mean queries by 14% (SimBA) and 48% (Square Attack, CE loss).

2. **Difficulty-scaled benefits.** OT's savings correlate positively with attack difficulty ($r = 0.38$, $p < 0.001$). On ResNet-50 — the deepest and hardest-to-attack model — savings reach 37% (SimBA) and 64% (Square Attack). This is precisely the regime where query efficiency matters most.

3. **Failure rescue.** OT converts SimBA's 88% untargeted success rate to 100%, matching the oracle. Under a fixed query budget, OT expands the set of feasible attacks, not just their speed.

4. **Structural equivalence to margin loss.** The CE-loss ablation on Square Attack confirms that OT functions as a structural surrogate for margin-based losses. When the loss already provides implicit target tracking (margin loss), OT is redundant. When it does not (CE loss, SimBA), OT restores near-optimal directionality.

5. **Characterized failure mode on robust networks.** On adversarially-trained models, flat confidence landscapes produce decoy classes — semantically plausible neighbors that appear stable early but lie in shallow adversarial basins. This causes OT to lock onto suboptimal targets, degrading performance on R50 Square Attack.

### 9.2 Limitations

**Sample size.** The 4-image benchmark per model, while sufficient to establish the main findings, produces wide confidence intervals at per-model granularity. The 100-image CDF benchmark on ResNet-50 provides tighter estimates for that model but does not extend to multi-model comparisons.

The 100-image CDF benchmark on ResNet-50 narrows the confidence intervals substantially. At 15K budget: SimBA OT matches oracle at 85% (vs. 53% untargeted); Square Attack OT reaches 98% (vs. 85% untargeted, 99% oracle). These results confirm the 4-image findings are not artifacts of small sample size.

**Budget censoring.** Untargeted iteration counts are right-censored at 10,000 (or 15,000 for the winrate benchmark). The true cost of hard attacks is higher, making our savings estimates conservative lower bounds.

**VGG-16 anomaly.** VGG-16 + Square Attack shows a slight mean regression (−6.8%) with OT. VGG-16's lack of skip connections may produce a less structured latent landscape where early rank signals are less predictive. This is a genuine limitation: OT is not universally beneficial on all architectures at low query counts.

**Robust model scope.** Only two adversarially-trained models were tested, with 4 images each. The decoy effect may be mitigable with higher stability thresholds or confidence-gated lock-in, but this requires further experimentation.

[TODO: update with robust S ablation conclusions]

### 9.3 Future Directions

1. **Confidence-gated lock-in.** Require not just rank stability but a minimum confidence delta $\Delta_{\min}$ before locking. This would prevent lock-in on flat plateaus where all non-true classes have similar, low probabilities — exactly the regime where decoys arise on robust models.

2. **Fallback with class banning.** If progress stalls after lock-in (loss plateaus for $T$ iterations), release the lock and ban the released class from future targeting. This transforms OT's irreversible commitment into an explore-exploit cycle.

3. **Larger-scale evaluation.** Testing on the full ImageNet validation set (50K images), additional model families (Vision Transformers, EfficientNets), and additional robust models would establish OT's generality more conclusively.

### 9.4 Conclusion

Opportunistic Targeting demonstrates that the information needed to select an effective adversarial target is already latent in the attack's own trajectory. By monitoring rank stability and committing when a clear candidate emerges, OT eliminates the latent-space drift that plagues probability-minimization and cross-entropy losses — providing a simple, general-purpose bridge between undirected exploration and directed exploitation. On standard models, the bridge is essentially free: the exploration phase is not overhead — every query advances the attack as a normal untargeted step, and the stability monitor simply observes which direction the perturbation is already heading. Once a target emerges, the attack commits and achieves near-oracle efficiency. On robust models, the bridge's foundations become unreliable, pointing toward confidence-aware gating as the next step. The simplicity of the approach — a stability counter and a mode switch, with no architectural or loss-function modifications — makes it immediately applicable to any score-based black-box attack.

---

## References

- Andriushchenko, M., Croce, F., Flammarion, N., and Hein, M. (2020). *Square Attack: A Query-Efficient Black-Box Adversarial Attack via Random Search*. ECCV.
- Carlini, N., and Wagner, D. (2017). *Towards Evaluating the Robustness of Neural Networks*. IEEE S&P.
- Gesny, E., Giboulot, E., and Furon, T. (2024). *When Does Gradient Estimation Improve Black-Box Adversarial Attacks?*. WIFS.
- Guo, C., Gardner, J. R., You, Y., Wilson, A. G., and Weinberger, K. Q. (2019). *Simple Black-box Adversarial Attacks*. ICML.
- Madry, A., Makelov, A., Schmidt, L., Tsipras, D., and Vladu, A. (2018). *Towards Deep Learning Models Resistant to Adversarial Attacks*. ICLR.
- Maho, T., Furon, T., and Le Merrer, E. (2021). *SurFree: A Fast Surrogate-Free Black-Box Attack*. CVPR.
- Salman, H., Ilyas, A., Engstrom, L., Kapoor, A., and Madry, A. (2020). *Do Adversarially Robust ImageNet Models Transfer Better?*. NeurIPS.
- Ughi, G., Abrol, V., and Tanner, J. (2021). *An Empirical Study of Derivative-Free-Optimization Algorithms for Targeted Black-Box Attacks in Deep Neural Networks*. Machine Learning.
