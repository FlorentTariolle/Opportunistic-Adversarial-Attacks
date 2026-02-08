# Black-Box Attack Objectives and Constraints

## 1. SimBA (Simple Black-box Adversarial Attacks)
**Paper:** [*Simple Black-box Adversarial Attacks*](https://proceedings.mlr.press/v97/guo19a/guo19a.pdf)

SimBA operates by iteratively adding or subtracting a vector from a predefined orthonormal basis to the image, accepting the change if it improves the adversarial loss.

### Loss Functions (Objectives)
* **Untargeted:** Minimize the probability of the correct class $y$.
    * $$\text{Loss} = P(y|x)$$
    * *Goal:* Decrease the confidence of the ground truth label until a misclassification occurs.
* **Targeted:** Maximize the probability of the target class $t$.
    * $$\text{Loss} = -P(t|x)$$
    * *Goal:* Increase the confidence of the target label.

### Budgets & Constraints
* **Query Budget ($Q_{max}$):**
    * Untargeted: **10,000 queries** (standard for ImageNet/CIFAR).
    * Targeted: **30,000 queries** (paper recommendation).
    * **Our benchmark:** Uses a **uniform 10,000 query budget** across all modes (untargeted, targeted-oracle, opportunistic) for fair iteration-count comparison. See `STANDARD_NN_BENCHMARK.md` for results.
* **Perturbation Budget:**
    * Controlled by the step size $\epsilon$ (default **0.03** in our $L_\infty$ implementation) and the number of iterations. Note: the original paper uses an $L_2$ setting ($\epsilon = 0.2$, norm bounded by $\sqrt{T} \cdot \epsilon$ where $T$ is the number of successful steps), which does not apply to our $L_\infty$ demo.

---

## 2. Square Attack
**Paper:** [*Square Attack: a query-efficient black-box adversarial attack via random search*](https://arxiv.org/abs/1912.00049) (Andriushchenko et al., ECCV 2020)

Square Attack is a score-based method based on random search that samples updates at the vertices of the $L_\infty$ neighborhood.

### Loss Functions (Objectives)
Unlike SimBA which often uses raw probabilities, Square Attack typically uses a **margin-based loss** to avoid saturation and provide a better gradient signal.

* **Untargeted:** Minimize the margin between the correct class and the nearest competitor.
    * $$\text{Loss}(x, y) = f_y(x) - \max_{k \neq y} f_k(x)$$
* **Targeted:** Minimize the margin between the highest non-target class and the target class $t$.
    * $$\text{Loss}(x, t) = \max_{k \neq t} f_k(x) - f_t(x)$$

### Budgets & Constraints
* **Query Budget ($Q_{max}$):**
    * Standard Benchmarks: **10,000 queries** (Untargeted), **100,000 queries** (Targeted).
* **Perturbation Constraint:**
    * Strictly enforces the $L_\infty$ norm constraint (e.g., $\epsilon = 8/255$ for ImageNet).
    * Updates are always projected back onto the $L_\infty$ ball.

---

## 3. Demonstrator & Benchmark Settings
**Standardization to $L_\infty$ Norm**

To ensure a fair and consistent comparison across all algorithms, we enforce the **$L_\infty$ (Chebyshev distance)** constraint for all generated adversarial examples.

* **Why $L_\infty$?**
    While some algorithms (like SimBA) are originally formulated or often evaluated using the $L_2$ norm, the $L_\infty$ norm provides a stricter guarantee on **individual pixel changes**. This ensures that no single pixel is perturbed beyond a fixed limit $\epsilon$, making the noise more uniformly imperceptible to the human eye compared to $L_2$, which allows for large spikes in specific pixels.

* **Implementation:**
    * **Square Attack:** Naturally supports and operates within $L_\infty$ bounds.
    * **SimBA:** The paper proposes two variants—Pixel Space (Cartesian basis) and DCT Space (low-frequency Discrete Cosine Transform basis). The demo uses **SimBA-DCT** by default, which restricts perturbations to low-frequency directions and is more query-efficient. Both variants operate under the same $\epsilon$ budget as other attacks.

* **Unified Input Space:**
    All models (standard torchvision and RobustBench) are wrapped to accept **[0, 1] pixel-space** input. Standard models use a `NormalizedModel` wrapper that applies ImageNet normalization internally. This means **$\epsilon$ is directly comparable across all attacks and models** — it always represents a perturbation in [0, 1] pixel space.

* **Global Parameters:**
    * **$\epsilon$:** Default $8/255 \approx 0.031$ in [0, 1] pixel space (standard ImageNet $L_\infty$ benchmark setting). The demo slider selects $n/255$ with $n \in [2, 64]$.
    * **Iteration Budget:** Default **10,000 iterations** for both attacks. Note: for Square Attack, the budget is not merely an early-stop ceiling — it controls the patch size schedule (see `SQUARE_ATTACK.md` Section 2b). A fixed budget is therefore essential for comparable iteration counts.
