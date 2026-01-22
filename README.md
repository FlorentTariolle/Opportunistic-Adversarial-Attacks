# Adversarial-BlackBox-Demo

Interactive web demo for generating adversarial examples that fool image classification models using black-box attacks (SimBA).

## What is this?

Upload an image and generate imperceptible perturbations that cause neural networks to misclassify it (untargeted attack), demonstrating the vulnerability of deep learning models to adversarial attacks.

## How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Launch the demo:
```bash
python launch_demo.py
```

3. Open your browser to `http://127.0.0.1:7860`

**Optional flags:**
- `--share` - Create a public sharing link
- `--host HOST` - Set server host (default: 127.0.0.1)
- `--port PORT` - Set server port (default: 7860)
