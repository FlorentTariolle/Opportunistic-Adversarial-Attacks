# Experimental Validation: Opportunistic Targeting on Square Attack

## 1. Objective
To isolate the efficiency gains provided by the **Opportunistic Targeting** framework (Rank-Stability Heuristic) and determine its relationship with different adversarial loss functions. Square Attack was selected as the testbed due to its ability to function with both **Margin-based** and **Cross-Entropy-based** losses.

## 2. Theoretical Framework: The "Drift" Hypothesis
We posit that standard Untargeted attacks utilizing simple probability minimization (e.g., Cross-Entropy) suffer from **Latent Space Drift**.
* **Drift:** Minimizing $P(y_{true})$ without a specific target forces the adversarial example to "run away" from the ground truth without a clear destination, resulting in an inefficient random walk.
* **Guidance:** Margin-based losses (like C&W) inherently include a target term ($\max P_{other}$), providing implicit directionality.

**Hypothesis:** *Opportunistic Targeting acts as an external guidance system, artificially imposing directionality on drift-prone loss functions, thereby restoring query efficiency.*

## 3. Experimental Results

We evaluated the framework on Square Attack ($L_\infty$) under two distinct loss configurations.

### A. Square Attack + Margin Loss (Standard)
* **Loss Function:** $L = P(y_{true}) - \max_{y \neq true} P(y)$
* **Observation:** The Opportunistic Strategy yielded **no significant reduction** in queries compared to the standard Untargeted mode.
* **Analysis:** The `max` term in the Margin Loss dynamically identifies the nearest decision boundary at *every iteration*. Consequently, the standard algorithm naturally performs a "dynamic targeted attack." The Opportunistic framework is redundant in this scenario as the loss function already mathematically encodes the optimal path.

### B. Square Attack + Cross-Entropy Loss (Ablation Study)
* **Loss Function:** $L = - \log(1 - P(y_{true}))$ (Solely minimizing ground truth probability).
* **Observation:** Without the guidance of the Margin term, the standard attack efficiency degrades significantly due to drift. The Opportunistic framework successfully corrected this behavior.

**Query Budget Comparison:**

| Mode | Iterations to Success | Efficiency vs Oracle | Notes |
| :--- | :--- | :--- | :--- |
| **Baseline (Untargeted)** | **~600** | 50% | High drift; inefficient random walk. |
| **Oracle (Targeted)** | **~300** | 100% | Targeted attack on the optimal class (known a priori). |
| **Opportunistic (Ours)** | **~310** | **97%** | **Near-optimal convergence.** |

*Note: The "Opportunistic" cost includes the ~10 iterations required for the Rank-Stability Heuristic to identify the target before locking on.*

## 4. Conclusion & Impact

1.  **Validation of Efficiency:** The Opportunistic Targeting framework restored **97% of the theoretical optimal efficiency** in the Cross-Entropy scenario, reducing the query count by nearly **50%** compared to the untargeted baseline (600 $\to$ 310).
2.  **Structural Surrogate:** The framework functions as a **structural surrogate for Margin Loss**. It provides necessary directionality to loss functions or attack vectors (like SimBA) that rely on simple probability minimization.
3.  **Generalizability:** These findings confirm that Opportunistic Targeting is a critical enhancement for attacks that lack intrinsic boundary-targeting logic, effectively bridging the gap between "dumb" optimization (Drift) and "smart" optimization (Targeted).