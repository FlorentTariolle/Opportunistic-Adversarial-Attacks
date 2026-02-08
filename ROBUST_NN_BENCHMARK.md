# Benchmark Analysis: Opportunistic Targeting on Robust Neural Networks

## 1. Experimental Setup

### Protocol

Same three-mode protocol as the standard benchmark (see `STANDARD_NN_BENCHMARK.md`):

| Mode | Description |
|------|-------------|
| **Untargeted** | Vanilla attack with no directional guidance. |
| **Targeted (oracle)** | Upper bound: target class chosen *a posteriori* from the untargeted result. |
| **Opportunistic (OT)** | Our method: lock onto the leading non-true class once rank-stability threshold is reached. |

**Configuration.** L-infinity budget epsilon = 8/255, query budget 10,000, three seeds (0, 1, 2). Stability threshold = **10** (doubled from the standard benchmark's 5, to account for robust models' flatter confidence landscape).

### Models

Two adversarially-trained ImageNet classifiers from RobustBench (Salman et al., 2020):

| Model | Architecture | Adversarial Training | Clean Acc. | Robust Acc. (L-inf 4/255) |
|-------|-------------|---------------------|------------|--------------------------|
| Salman2020Do_R18 | ResNet-18 | PGD adversarial training | 52.5% | 25.3% |
| Salman2020Do_R50 | ResNet-50 | PGD adversarial training | 56.3% | 27.5% |

### Images

Same four images: `corgi.jpg`, `porsche.jpg`, `dumbbell.jpg`, `hammer.jpg`. Unlike the standard benchmark, **no all-fail filtering is applied** — failed runs carry meaningful progress metrics (margin, confusion gain) that are central to the analysis. This yields **144 runs** (48 per mode).

### Epsilon and Budget

We use **epsilon = 8/255**, consistent with the standard benchmark. While the RobustBench ImageNet L-inf leaderboard evaluates at 4/255, we retain 8/255 to enable direct comparison across our standard and robust benchmarks under identical perturbation budgets. Note that these models were adversarially trained at epsilon = 4/255 (Madry et al., 2018), so our attack budget exceeds the training radius.

The **query budget of 10,000** matches the standard benchmark and the original SimBA and Square Attack papers' default for standard models. This is likely insufficient for robust models — the Square Attack paper recommends 20,000 queries with random restarts for robust evaluation (Andriushchenko et al., 2020). Our budget is constrained by single-GPU compute. The low success rates reported below should be interpreted with this limitation in mind: a larger query budget would likely improve absolute success rates for all modes, though the *relative* comparison between modes (which is our focus) should be less affected.

### Key Difference from Standard Benchmark

On standard networks, most attacks succeed within budget and iteration count is the primary metric. On robust networks, **most attacks fail** (overall success rate: 12.5–37.5%). Iteration counts are therefore uninformative. Instead, we report **final margin** — the gap between true-class and best-other-class confidence at attack termination:

```
margin = max(P(true_class) - P(best_other_class), 0)
```

Lower margin = model closer to misclassification = better attack. This is visually consistent with the standard benchmark where lower iteration bars = better.

---

## 2. Success Rates

| Model | Method | Untargeted | Targeted (oracle) | Opportunistic |
|-------|--------|-----------|-------------------|---------------|
| R18 | SimBA | 25.0% | 25.0% | 25.0% |
| R18 | Square Attack | 50.0% | 50.0% | 50.0% |
| R50 | SimBA | 0.0% | 0.0% | 0.0% |
| R50 | Square Attack | 25.0% | 25.0% | **0.0%** |

Success rates are dramatically lower than on standard networks (where OT achieved 100%). Notably:

- **SimBA is completely mode-insensitive** on robust models: all three modes produce identical success rates. This contrasts sharply with standard networks, where OT rescued SimBA's 88.1% untargeted rate to 100%.
- **R50 Square Attack opportunistic drops to 0%** — the only case where OT actively hurts. This is the central finding requiring explanation (Section 4).

---

## 3. Final Margin Analysis

Since most attacks fail, we compare mean final margin across all runs (lower = better):

![Final Margin Per Model](results/figures/robust/fig_margin_per_model.png)

| Model | Method | Untargeted | Targeted | Opportunistic |
|-------|--------|-----------|----------|---------------|
| R18 | SimBA | 0.330 | 0.333 | 0.352 |
| R18 | Square Attack | **0.120** | **0.077** | **0.075** |
| R50 | SimBA | 0.568 | 0.576 | 0.576 |
| R50 | Square Attack | **0.221** | **0.188** | **0.267** |

### SimBA: Mode-Invariant on Robust Models

SimBA's margin is virtually identical across all three modes for both models. On R18, margin hovers around 0.33–0.35; on R50, around 0.57. The targeting mode simply does not matter — SimBA's coordinate-wise perturbations lack the power to exploit directional guidance against adversarially-trained models. This is consistent with recent findings that *all* query-based black-box attacks struggle against even simple adversarially-trained models, achieving < 4% ASR in systematic evaluations (Djilani et al., 2024). This stands in stark contrast to standard networks where SimBA saw 14.3% iteration savings with OT.

### Square Attack: Targeting Helps on R18, Hurts on R50

On R18, targeting (both oracle and opportunistic) reduces the margin from 0.120 to ~0.076 — a meaningful improvement. Opportunistic matches oracle performance almost exactly, suggesting the stability heuristic locks onto viable targets for this model.

On R50, **opportunistic produces worse margin (0.267) than untargeted (0.221)**. Targeted oracle still helps (0.188), meaning the problem is not with targeting itself but with *which class* OT locks onto. This points to a target selection failure specific to R50.

---

## 4. The Decoy Hypothesis: Why OT Fails on R50

### 4.1 Lock-Match Rates Are Poor

We compare the class OT locks onto vs. the final predicted class of the untargeted attack:

![Lock-Match Rates](results/figures/robust/fig_lock_match.png)

| Model | Method | Lock-Match Rate |
|-------|--------|----------------|
| R18 | SimBA | 25.0% |
| R18 | Square Attack | 33.3% |
| R50 | SimBA | 0.0% |
| R50 | Square Attack | 0.0% |

On standard networks, lock-match rates were 78–84% (dashed reference line). On robust networks, they collapse to 0–33%. However, this metric is partially an artifact: for failed untargeted attacks, the final predicted class is simply the **true class** (the model never flipped), so no non-true locked class can ever match. A fairer comparison using peak adversarial class (Section 4.2) shows that OT often identifies the correct target — the attacks just fail regardless.

### 4.2 Semantically Plausible Decoys

The locked classes are not random — they are semantically related neighbors that appear stable in early exploration. This is consistent with Ozbulak et al. (2021), who found that 71% of adversarial misclassifications on ImageNet land in semantically similar classes within the WordNet hierarchy. We compare them to the **peak adversarial class** (the non-true class reaching highest confidence during the untargeted attack), which is the class oracle targeting aims at:

| Image | True Class | OT Locks Onto | Peak Adversarial Class |
|-------|-----------|---------------|----------------------|
| corgi.jpg | Pembroke (263) | Cardigan (264) | Dingo (273) / Cardigan (264) |
| dumbbell.jpg | Dumbbell (543) | Barbell (422) | Barbell (422) |
| hammer.jpg | Hammer (587) | Hatchet (596) | Hatchet (596) / **Tripod (872)** |
| porsche.jpg | Sports car (817) | Racer (751) | Racer (751) |

For dumbbell and porsche, OT correctly locks onto the same class as the oracle target — but the attack still fails because even correctly-targeted perturbations cannot overcome the defense within budget. The critical failure is on **hammer.jpg**: the peak adversarial class is model- and method-dependent. R50 Square Attack's untargeted mode naturally converges to tripod (872), but OT locks onto hatchet (596) — a semantically plausible neighbor that appears stable early on but lies in a shallower adversarial basin. We call these early-stable-but-ultimately-unviable classes **decoys**.

### 4.3 Per-Image Margin Heatmap

The heatmap below shows final margin for every (model, method, image, mode) cell. Green = low margin (attack succeeded or came close), red = high margin (attack made little progress). The decoy effect is visible as cells where opportunistic is lighter (worse) than untargeted:

![Margin Heatmap](results/figures/robust/fig_margin_heatmap.png)

Key observations: SimBA columns are nearly uniform across modes (mode-invariant). SquareAttack shows meaningful mode differences, particularly in the bottom-right panel (R50) where hammer.jpg goes from green (0.000) to yellow (0.182) under opportunistic.

### 4.4 The Smoking Gun: R50 Square Attack on hammer.jpg

This is the only configuration where opportunistic targeting causes a success rate *degradation*:

| Mode | Margin | Success | Target/Locked Class |
|------|--------|---------|-------------------|
| Untargeted | 0.000 | 3/3 | Converges to Tripod (872) |
| Targeted (oracle) | 0.000 | 3/3 | Target = Tripod (872) |
| Opportunistic | 0.182 | **0/3** | Locks onto Hatchet (596) |

The untargeted attack naturally drifts toward class 872 (tripod) and succeeds. The oracle knows this and targets 872 directly. But the stability heuristic locks onto 596 (hatchet) — a semantically plausible neighbor that appears stable early in exploration. Once locked, the attack wastes its remaining ~9,900 queries pushing toward a class that lies in a shallow adversarial basin, never reaching misclassification. The per-seed variance within this cell is near zero (std < 0.01), confirming the failure is structural, not stochastic.

### 4.5 Why R18 Survives

On R18, OT locks onto the correct peak adversarial class (barbell for dumbbell, racer for porsche), and the model's lower robust accuracy means targeting provides a genuine margin improvement (Section 3: R18 Square Attack margin drops from 0.120 to 0.075 with OT). The defense is weaker, so even incomplete convergence toward the right target reduces margin meaningfully. On R50, the defense is stronger, and the one case where OT picks the *wrong* peak class (hammer.jpg: hatchet instead of tripod) is exactly the case that causes the success rate regression from 100% to 0%.

---

## 5. Lock-in Timing

| Model | Method | Mean Switch | Median | Range |
|-------|--------|------------|--------|-------|
| R18 | SimBA | 12.5 | 12 | 11–16 |
| R18 | Square Attack | 225.1 | 248 | 69–316 |
| R50 | SimBA | 12.2 | 12 | 11–14 |
| R50 | Square Attack | 109.2 | 94 | 48–236 |

SimBA locks in ~12 iterations regardless of model — essentially the stability threshold plus one. With threshold = 10, this means the locked class is determined by the model's initial confidence landscape, not the attack's optimization progress. On flat robust landscapes, this near-instant lock makes decoy classes particularly dangerous.

Square Attack locks later (94–248 iterations), giving the confidence landscape more time to differentiate, but still too early for the correct basin to emerge on R50.

---

## 6. Discussion

### OT's Effectiveness Depends on Landscape Geometry

On standard networks, the loss landscape is peaked and adversarial basins are deep. The first class to stabilize during exploration is overwhelmingly the "right" one (78–84% match rate). On robust networks, adversarial training smooths the input loss landscape (Tsipras et al., 2019) and reduces prediction overconfidence (Grabinski et al., 2022), creating an environment where multiple competing classes have similar early stability. Schwinn et al. (2023) independently found that "previous attacks under-explore the perturbation space during optimization, which leads to unsuccessful attacks for samples where the initial gradient direction is not a good approximation of the final adversarial perturbation direction" — essentially the same phenomenon our decoy hypothesis describes. The stability heuristic cannot distinguish deep basins (viable targets) from shallow ones (decoys).

### Potential Mitigations

1. **Higher stability threshold**: The current threshold of 10 may be too low for robust models. A threshold of 50–100 could allow more of the landscape to reveal itself before committing. However, this delays lock-in and reduces the query budget advantage.

2. **Confidence-gated lock-in**: Require not just rank stability but a minimum confidence delta before locking. This would prevent locking on flat plateaus where all non-true classes have similar, low probabilities.

3. **Fallback**: If progress stalls, release the lock. (And we ban the released class from potential targets)

These are open directions — the current benchmark is too small (2 models, 4 images) to validate any mitigation rigorously.

### Sample Size Caveat

With 12 runs per (model, method, mode) cell and only 2 robust models, we are highly sensitive to individual image effects. The R50 Square Attack result is driven almost entirely by hammer.jpg. A larger benchmark (more models, more images) is needed to determine whether the decoy effect is systematic or specific to this configuration.

---

## 7. Summary

1. **SimBA is mode-invariant on robust models.** Targeting mode has no measurable effect on margin or success rate. SimBA lacks the perturbation power to exploit directional guidance against adversarial training.

2. **Square Attack shows mixed results.** On R18, opportunistic targeting matches oracle performance (margin 0.075 vs 0.077). On R50, it underperforms untargeted (margin 0.267 vs 0.221) due to locking onto a semantically plausible but ultimately unviable target class.

3. **The decoy hypothesis explains OT's critical failure mode.** OT often identifies the correct peak adversarial class, but when the peak class is model- and method-dependent (hammer.jpg), OT can lock onto an early-stable neighbor (hatchet) while the untargeted attack naturally converges to a different, viable class (tripod). This decoy effect caused a success rate regression from 100% to 0% on R50 Square Attack.

4. **OT is not reliably beneficial on robust networks.** Unlike standard models where OT was a strict improvement on hard attacks, the robust regime presents a fundamentally different challenge. The stability signal that drives OT's success on standard models becomes misleading when the confidence landscape is flat.

5. **The results are preliminary.** Two robust models and four images are insufficient to draw definitive conclusions. The decoy effect may be mitigable with threshold tuning or confidence-gated lock-in, but this requires further experimentation.

---

## 8. Figures Reference

| Figure | File | Description |
|--------|------|-------------|
| Final Margin Per Model | `fig_margin_per_model` | Mean final margin by mode, per model (lower = better) |
| Margin Heatmap | `fig_margin_heatmap` | Per-image margin breakdown showing decoy effect at cell level |
| Lock-Match Rate | `fig_lock_match` | Lock-match rate collapse vs. standard benchmark (~80% reference) |

All figures in `results/figures/robust/` as PNG (300 dpi) and PDF.

---

## 9. References

- Andriushchenko, M., Croce, F., Flammarion, N., & Hein, M. (2020). *Square Attack: a query-efficient black-box adversarial attack via random search.* ECCV 2020.
- Grabinski, J., et al. (2022). *Robust Models are less Over-Confident.* NeurIPS 2022.
- Guo, C., Gardner, J., You, Y., Wilson, A. G., & Weinberger, K. (2019). *Simple Black-box Adversarial Attacks.* ICML 2019.
- Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). *Towards Deep Learning Models Resistant to Adversarial Attacks.* ICLR 2018.
- Ozbulak, U., Pintor, M., Van Messem, A., & De Neve, W. (2021). *Evaluating Adversarial Attacks on ImageNet: A Reality Check on Misclassification Classes.* NeurIPS 2021 Workshop.
- Djilani, M., Ghamizi, S., & Cordy, M. (2024). *RobustBlack: Challenging Black-Box Adversarial Attacks on State-of-the-Art Defenses.* arXiv preprint arXiv:2412.20987.
- Salman, H., Ilyas, A., Engstrom, L., Kapoor, A., & Madry, A. (2020). *Do Adversarially Robust ImageNet Models Transfer Better?* NeurIPS 2020.
- Schwinn, L., et al. (2023). *Exploring Misclassifications of Robust Neural Networks to Enhance Adversarial Attacks.* Applied Intelligence, 53, 19843–19859.
- Tsipras, D., Santurkar, S., Engstrom, L., Turner, A., & Madry, A. (2019). *Robustness May Be at Odds with Accuracy.* ICLR 2019.

---

*Benchmark run on a single GPU, February 2026. Analysis script: `analyze_benchmark.py --source robust`.*
