# Roadmap

## Phase 1: Algorithm Generalization (Square Attack)
*Goal: Prove that "Opportunistic Targeting" is algorithm-agnostic and not just an artifact of SimBA.*

- [x] **Implement Square Attack:** Add standard Square Attack (Untargeted & Targeted modes) to the demonstrator.
- [x] **Integrate Opportunistic Logic:** Adapt the Rank-Stability wrapper to control Square Attack's switching mechanism.
- [x] **Validation:** Compare query efficiency ($Q$) of Opportunistic Square vs. Standard Untargeted Square.

## Phase 2: Robustness Validation (Hardened Models)
*Goal: Verify effectiveness against models with smoothed decision boundaries.*

- [ ] **RobustBench Integration:** Add support for loading robustly trained models (e.g., adversarial training).
- [ ] **Stress Testing:** Run Opportunistic attacks against hardened targets.
- [ ] **Analysis:** Determine if the "Path of Least Resistance" strategy remains effective in robust latent spaces where margins are maximized.
