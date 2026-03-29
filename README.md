# Learning Radar Waveforms: End-to-End Neural Generation via Differentiable Ambiguity Functions

Code and experiments for the paper *"Learning Radar Waveforms: End-to-End Neural Generation via Differentiable Ambiguity Functions"*, submitted to IEEE Transactions on Aerospace and Electronic Systems.

## Overview

This repository provides:

1. **Differentiable ambiguity function** — a PyTorch implementation of the periodic ambiguity function that is fully compatible with automatic differentiation, enabling gradient-based optimization with arbitrary loss functions.

2. **Neural waveform generator** — an MLP that takes design parameters (e.g., PSL/LPI trade-off weight) as input and outputs optimized constant-modulus radar waveforms. Trained end-to-end through the differentiable ambiguity function. Once trained, generates waveforms in <1 ms.

3. **Experimental validation** — multi-seed benchmarks (gradient vs GA), GPU scaling, spectral mask constraints, downstream detection experiments, and architecture ablations.

## Key Results

| Experiment | Result |
|---|---|
| Full 2D vs 1D autocorrelation | **7.3 dB PSL improvement** |
| Gradient vs GA (20 seeds) | **2.2–3.0 dB** improvement, **2.1×** speedup |
| Neural generator inference | **<1 ms** per waveform (vs ~7 s iterative) |
| GPU acceleration (Tesla T4) | Up to **39×** speedup at N=512 |
| Spectral mask constraint | 7.7% → 0.00% notch energy, 0.2 dB PSL cost |

## Quick Start

### Installation

```bash
git clone https://github.com/marcbara/graf-psl-lpi
cd graf-psl-lpi
python -m venv .venv
.venv/Scripts/activate   # Windows
# source .venv/bin/activate  # Linux/Mac
pip install torch numpy matplotlib seaborn pyyaml
```

### Run the differentiable ambiguity function

```python
import torch
from graf import DifferentiableAmbiguity, compute_psl

# Create a random phase-coded waveform
N = 256
phases = torch.rand(N) * 2 * 3.14159
s = torch.exp(1j * phases)

# Compute differentiable ambiguity function
amb = DifferentiableAmbiguity()
chi = amb(s)  # [N, N] tensor, fully differentiable

# Compute PSL and backpropagate
psl = compute_psl(chi)
psl.backward()  # gradients flow back to phases
```

### Train the neural waveform generator

```bash
python neural_waveform_generator.py
```

### Run all benchmarks

```bash
python benchmarks.py
```

### Run on GPU (Modal)

```bash
pip install modal
python run_modal.py       # Core experiments
python run_modal_v2.py    # Extended experiments (baseline, spectral mask, ablations)
```

### Run radar validation experiments

```bash
python radar_validation.py
```

### Generate paper figures

```bash
python generate_figures.py
```

## Repository Structure

```
.
├── graf.py                      # Differentiable ambiguity function + gradient/GA optimization
├── neural_waveform_generator.py # Neural waveform generator (train + evaluate)
├── benchmarks.py                # Multi-seed, multi-N, GPU benchmarks
├── radar_validation.py          # Downstream detection + interference experiments
├── experiments_v2.py            # Baseline comparison, spectral mask, ablations
├── generate_figures.py          # Generate all paper figures from results
├── run_modal.py                 # Modal cloud GPU runner (core experiments)
├── run_modal_v2.py              # Modal cloud GPU runner (extended experiments)
├── config.yaml                  # Experiment configuration
├── config_loader.py             # Configuration management
├── requirements.txt             # Python dependencies
│
├── IEEE_TAES/                   # Paper (IEEE TAES submission)
│   ├── main.tex                 # Full paper source
│   ├── references.bib           # Bibliography (~40 references)
│   ├── main.pdf                 # Compiled paper
│   └── figures/                 # Paper figures (PDF)
│
├── GRAF___IET_RSN/              # Previous submission (IET RSN, rejected)
│   ├── main.tex                 # Original paper
│   └── Reviewers.pdf            # Reviewer comments
│
└── results/                     # Experimental results (JSON)
    ├── multi_seed_N256.json     # 20-seed gradient vs GA comparison
    ├── neural_generator_N256.json
    ├── gpu_benchmark.json
    ├── scaling_benchmark.json
    ├── v2_all_results.json      # Baseline, spectral mask, ablations
    └── radar_validation.json    # Detection + interference results
```

## Method

### Differentiable Ambiguity Function

The periodic ambiguity function is reformulated as:

1. Construct circulant shift matrix **S**
2. Element-wise correlation: **R** = s ⊙ **S***
3. Column-wise FFT: **X** = FFT(**R**)
4. Squared magnitude: χ = |**X**|²

All operations are natively differentiable in PyTorch. Complexity: O(N² log N). Memory: O(N²).

### Neural Waveform Generator

- **Input:** design parameters [λ, noise_seed]
- **Network:** MLP with residual blocks, LayerNorm, GELU (332K parameters)
- **Output:** N phases → exp(jφ) → constant-modulus waveform
- **Training:** backpropagate PSL + spectral variance loss through the differentiable ambiguity function
- **Inference:** single forward pass, <1 ms

## Citation

```bibtex
@article{bara2026learning,
  title={Learning Radar Waveforms: End-to-End Neural Generation via Differentiable Ambiguity Functions},
  author={Bara, Marc},
  journal={IEEE Trans. Aerosp. Electron. Syst.},
  year={2026},
  note={Submitted}
}
```

## License

Academic and research use only. See [LICENSE](LICENSE) for details.

## Contact

Marc Bara Iniesta — marcoantonio.bara@esade.edu
