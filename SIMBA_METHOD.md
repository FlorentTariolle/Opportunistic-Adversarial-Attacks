# SimBA: Simple Black-box Adversarial Attacks

## Overview

SimBA (Simple Black-box Adversarial Attack) is a query-efficient black-box attack algorithm proposed by Guo et al. (ICML 2019). It constructs adversarial images using only the model's confidence scores, without requiring gradient access.

**Key insight:** If the distance to a decision boundary is small, we don't need to be precise about the direction—randomly sampling from an orthonormal basis and taking steps that reduce the target class confidence is sufficient.

---

## Algorithm

### Core Principle

SimBA iteratively perturbs an image by:
1. Randomly selecting a direction **q** from a predefined orthonormal basis **Q**
2. Trying to add **+εq** to the image
3. If this reduces the true class probability, accept the step
4. Otherwise, try **-εq**
5. Repeat until misclassification or budget exhausted

### Pseudocode (from paper)

```
procedure SIMBA(x, y, Q, ε):
    δ = 0
    p = P(y | x)
    while p_y = max_{y'} p_{y'}:
        Pick randomly without replacement: q ∈ Q
        for α ∈ {ε, -ε}:
            p' = P(y | x + δ + αq)
            if p'_y < p_y:
                δ = δ + αq
                p = p'
                break
    return δ
```

This simple procedure requires only **1.4–1.5 queries per iteration** on average (since we first try +ε, and only query -ε if needed).

---

## Search Space Variants

SimBA supports two orthonormal bases for the search directions **Q**:

### 1. Pixel Space (Cartesian Basis) — SimBA

- **Basis:** Standard basis Q = I (identity matrix)
- **Each direction:** Perturbs a single pixel in a single color channel
- **Characteristics:**
  - Produces sparse perturbations (few pixels changed significantly)
  - Corresponds to an L₀-style attack
  - 73% of randomly sampled directions are descending

### 2. DCT Space (Discrete Cosine Basis) — SimBA-DCT

- **Basis:** Low-frequency components of the Discrete Cosine Transform
- **Each direction:** Perturbs a DCT coefficient, affecting a spatial block
- **Characteristics:**
  - Perturbations spread across all pixels (smoother noise)
  - More query-efficient: 98% of directions are descending
  - Exploits the finding that low-frequency noise is more likely to be adversarial

The paper recommends keeping only a fraction (e.g., 1/8) of the lowest frequency directions for improved efficiency.

---

## Loss Functions

### Untargeted Attack
Minimize the probability of the correct class:
$$\mathcal{L}(x, y) = P(y | x)$$

**Goal:** Decrease confidence until any misclassification occurs.

### Targeted Attack
Maximize the probability of the target class:
$$\mathcal{L}(x, t) = -P(t | x)$$

**Goal:** Increase confidence of target class t until it becomes the prediction.

---

## Perturbation Norm Bound

A key property of SimBA is its **tight L₂ norm bound** due to orthonormality.

After T successful updates (each adding ±εq):
$$\|\delta_T\|_2 = \sqrt{T} \cdot \epsilon$$

**Trade-off:** Larger ε means faster convergence but higher perturbation norm.

---

## Implementation Details

### Our Demo Implementation

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `epsilon` | 0.03 | Step size / max perturbation (L∞) |
| `max_iterations` | 1000 | Query budget per image |
| `use_dct` | True | Use DCT space (SimBA-DCT) |
| `block_size` | 8 | DCT block size (8×8) |

**File:** `src/attacks/simba.py`

### Key Implementation Choices

1. **Perturbation constraint:** Our implementation uses L∞ norm (each pixel bounded by ε), whereas the paper focuses on L₂ norm bounds.

2. **DCT implementation:**
   - Images divided into 8×8 blocks per channel
   - For 224×224×3 images: 28×28×3×64 = 150,528 candidate directions
   - Basis vectors computed on-the-fly for memory efficiency

3. **Acceptance criterion:** A perturbation is accepted if:
   - It causes misclassification (success), OR
   - It reduces the true class confidence

4. **Image normalization:** Uses standard ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).

### Code Structure

```
SimBA.generate(x, y)
    └── _attack_single_image(x, y)
            ├── _generate_dct_candidate_indices(x)  # or pixel indices
            ├── For each iteration:
            │   ├── _create_perturbation(shape, idx)
            │   │       └── _create_single_dct_basis_vector()
            │   │               └── _idct_2d()  # Inverse DCT
            │   ├── Try +perturbation → check confidence
            │   └── Try -perturbation → check confidence
            └── Return adversarial image
```

---

## Why SimBA Works

1. **Natural images are close to decision boundaries:** Most classifiers in high-dimensional spaces have decision boundaries near data points (Fawzi et al., 2018).

2. **Random search is sufficient:** When close to a boundary, many directions lead toward it—precise gradient estimation is unnecessary.

3. **Orthonormality prevents wasted effort:** Sampling without replacement from orthonormal vectors ensures:
   - No direction is tried twice
   - Directions don't cancel each other out
   - Norm grows predictably

4. **Low-frequency bias:** DCT space exploits the observation that adversarial perturbations are often more effective in low-frequency components.

---

## Comparison: SimBA vs SimBA-DCT

| Aspect | SimBA (Pixel) | SimBA-DCT |
|--------|---------------|-----------|
| Search space | Individual pixels | DCT coefficients |
| Perturbation pattern | Sparse, sharp | Smooth, spread out |
| Descending directions | ~73% | ~98% |
| Query efficiency | Good | Better (initially) |
| Success rate | Higher (~100%) | Slightly lower (~97%) |
| Best for | Guaranteed success | Fastest convergence |

**Trade-off:** SimBA-DCT converges faster but may fail on some images if restricted to low frequencies only. SimBA (pixel space) is slower but more robust.

---

## References

- **Paper:** Guo, C., Gardner, J. R., You, Y., Wilson, A. G., & Weinberger, K. Q. (2019). *Simple Black-box Adversarial Attacks*. ICML 2019.
- **Code:** https://github.com/cg563/simple-blackbox-attack
- **Our implementation:** `src/attacks/simba.py`
