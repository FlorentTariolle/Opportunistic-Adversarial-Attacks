# Opportunistic Target Selection

Early Directional Commitment for Query-Efficient Black-Box Adversarial Attacks

## Overview

**Opportunistic Target Selection (OTS)** is a lightweight wrapper that adds early target selection to any score-based black-box adversarial attack. It runs the attack in untargeted mode for a short exploration phase, then switches to a targeted objective against whichever non-true class currently leads. By acting as a margin-loss surrogate for attacks that lack implicit target tracking, OTS eliminates class drift without requiring architectural modification, gradient access, or a priori target-class knowledge.

See [`paper/main.tex`](paper/main.tex) for the full paper.

---

## Project Structure

```
src/                    Core library
  attacks/              Attack implementations (SimBA, SquareAttack, Bandits)
  models/               Model loaders (torchvision, RobustBench)
  utils/                Image preprocessing & visualization
  demo/                 Gradio demonstrator app
demo/                   Demo entry point
benchmarks/             Benchmark scripts (generate CSV results)
analysis/               Analysis scripts (generate figures from CSVs)
slurm/                  HPC job scripts (CRIANN Arctic)
paper/                  LaTeX paper
results/                Benchmark CSVs and figures
data/                   ImageNet class index and demo images
```

---

## Quick Start

1. **Install dependencies**

    ```bash
    pip install -r requirements-gpu.txt   # With GPU (NVIDIA CUDA)
    pip install -r requirements-cpu.txt   # CPU only
    ```

2. **Launch the demonstrator**

    ```bash
    python demo/launch.py
    ```

3. **Access the interface**
    Open [http://127.0.0.1:7860](http://127.0.0.1:7860) in your browser.

---

## Benchmarks

| Script | Description |
|--------|-------------|
| `benchmarks/benchmark.py` | Multi-model benchmark: 5 standard models + 2 robust models, 3 attacks, 3 modes |
| `benchmarks/winrate.py` | ResNet-50 CDF benchmark: 100 images, 15K budget, bootstrapped CI |
| `benchmarks/ablation_s.py` | Stability threshold sweep S={2..15} on standard ResNet-50 |
| `benchmarks/ablation_s_robust.py` | Stability threshold sweep on robust ResNet-50 |
| `benchmarks/ablation_naive.py` | Naive fixed-iteration switching vs OTS (standard and robust) |
| `benchmarks/margin.py` | Margin vs CE loss ablation on SquareAttack |
| `benchmarks/landscape.py` | Per-iteration confidence history collection |
| `benchmarks/theta.py` | Perturbation alignment with oracle direction |

```bash
python benchmarks/benchmark.py
python benchmarks/winrate.py
```

## Analysis

Regenerate figures from benchmark CSVs:

```bash
python analysis/analyze_benchmark.py
python analysis/analyze_winrate.py
python analysis/analyze_ablation_s.py
python analysis/analyze_ablation_naive.py
python analysis/analyze_margin.py
python analysis/analyze_lockmatch.py
python analysis/analyze_oracle_beat.py
```
