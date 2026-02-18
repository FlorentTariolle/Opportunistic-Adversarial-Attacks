# Opportunistic-Adversarial-Attacks

A Rank-Stability Heuristic for Query-Efficient Black-Box Adversarial Attacks

## Overview

**Opportunistic Targeting (OT)** is a lightweight wrapper that adds dynamic target selection to any score-based black-box adversarial attack. It monitors the rank stability of the leading non-true class during an untargeted attack and switches to a targeted objective once a stable candidate emerges. OT requires no architectural modification, no gradient access, and no a priori target-class knowledge.

See [`paper_draft.md`](paper_draft.md) for the full paper.

---

## Quick Start

1. **Install dependencies**

    **With GPU (NVIDIA CUDA 12.1):**

    ```bash
    pip install -r requirements-gpu.txt
    ```

    **CPU only:**

    ```bash
    pip install -r requirements-cpu.txt
    ```

2. **Launch the demonstrator**

    ```bash
    python launch_demo.py
    ```

3. **Access the interface**
    Open [http://127.0.0.1:7860](http://127.0.0.1:7860) in your browser.

---

## Benchmarks

**Multi-model benchmark** (`benchmark.py`): evaluates OT across 4 standard ImageNet classifiers (AlexNet, ResNet-18, VGG-16, ResNet-50) and 2 adversarially-trained models, with 3 seeds per configuration.

**ResNet-50 in-depth benchmark** (`benchmark_winrate.py`): 100-image evaluation on ResNet-50 with bootstrapped CDF curves.

**Stability threshold ablation** (`benchmark_ablation_s.py`): sweeps $S \in \{2, 3, 5, 8, 10, 12, 15\}$ on ResNet-50.

**Theta convergence** (`benchmark_theta.py`): computes perturbation alignment with oracle direction over 100 images on ResNet-50.

```bash
python benchmark.py
python benchmark_winrate.py
python benchmark_ablation_s.py
python benchmark_theta.py
```

Regenerate figures from benchmark CSVs:

```bash
python analyze_benchmark.py
python analyze_winrate.py
python analyze_ablation_s.py
```
