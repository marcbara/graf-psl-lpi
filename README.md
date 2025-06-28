# GRAF: Gradient-based Radar Ambiguity Functions

PyTorch implementation of GRAF from the paper "Differentiable Radar Ambiguity Functions: Mathematical Formulation and Computational Implementation"

## About GRAF
GRAF is a general mathematical framework that enables gradient-based optimization of radar waveforms through differentiable ambiguity functions. This repository demonstrates GRAF through PSL-LPI optimization experiments, showcasing the framework's capabilities for multi-objective radar waveform design.

## Other Implementations
- TensorFlow: [Coming soon / Community implementations welcome]
- JAX: [Community implementations welcome]

## Key Features

🎯 **Differentiable Ambiguity Functions**: PyTorch-based implementation enabling gradient-based radar waveform optimization

⚡ **Multiple Optimization Methods**: 
- GRAF (Gradient-based) - Novel differentiable approach
- Genetic Algorithm baseline for comparison

🔄 **Flexible Multi-Objective Framework**: Demonstrated with PSL-LPI optimization:
- **PSL** (Peak Sidelobe Level) - Radar performance  
- **LPI** (Low Probability of Intercept) - Stealth capability
- **Extensible** to any radar performance metrics

📊 **Comprehensive Analysis**: Pareto frontier exploration showing fundamental PSL vs LPI trade-offs

⚙️ **Configurable Framework**: YAML-based configuration system with quick/full execution modes

## Example Application: PSL-LPI Trade-off Study

This repository demonstrates GRAF through a comprehensive PSL vs LPI optimization study, but the framework supports optimization of any differentiable radar metrics.

## Quick Start

### Installation
```bash
git clone https://github.com/marcbara/graf-psl-lpi
cd graf
pip install -r requirements.txt
```

### Basic Usage
```bash
# Quick test (N=64, ~2 minutes)
python graf.py

# Switch to full mode in config.yaml:
# execution_mode: "full"  # N=256, ~15 minutes
```

### Configuration Modes

| Mode | Sequence Length | Iterations | Generations | Lambda Points | Runtime |
|------|----------------|------------|-------------|---------------|---------|
| **quick** | N=64 | 200 | 100 | 4 | ~2 min |
| **full** | N=256 | 2000 | 300 | 5 | ~15 min |

## Results

GRAF generates three main outputs:

1. **`fig1_main_comparison.png`** - Convergence analysis and Pareto frontier
2. **`fig2_performance_summary.png`** - Computational efficiency and PSL achievements  
3. **`fig3_spectrum_comparison.png`** - Spectral analysis of optimized waveforms

### Key Findings

- **Speed**: GRAF achieves 2× faster convergence than genetic algorithms
- **Performance**: Comparable or superior PSL levels across all λ values
- **Trade-offs**: Clear Pareto frontier between radar performance and stealth

## Method Overview

### GRAF Algorithm
```python
# Core GRAF optimization loop
for iteration in range(max_iterations):
    s = torch.exp(1j * phases)           # Generate complex waveform
    chi = ambiguity_function(s)          # Compute differentiable ambiguity
    loss = psl_loss + λ * lpi_loss      # Multi-objective loss
    loss.backward()                      # Gradient computation
    optimizer.step()                     # Parameter update
```

### Configuration System
Edit `config.yaml` to control all experimental parameters:

```yaml
execution_mode: "quick"  # or "full"

signal:
  sequence_length: 64    # Radar waveform length
  mainlobe_width: 3      # PSL computation parameter

loss:
  lpi_scale_factor: 2000 # Balance PSL vs LPI objectives
```

## Academic Usage

### Citation
```bibtex
@article{BARA2025GRAF,
  title={Differentiable Radar Ambiguity Functions: Mathematical Formulation and Computational Implementation},
  author={Marc Bara Iniesta},
  journal={TBD},
  year={2025},
  note={Patent Pending}
}
```

### Patent Reference
GRAF methodology is protected by provisional patent application filed [Date]. Please cite both the academic paper and patent status when referencing this work.

### Reproducing Paper Results
1. Clone repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run full experiment: Set `execution_mode: "full"` in `config.yaml`
4. Execute: `python graf.py`
5. Results saved to `results/` directory

## GRAF Framework Extensibility

While this repository focuses on PSL-LPI optimization, GRAF supports any differentiable radar metrics:

**Potential Applications:**
- PSL vs Power Consumption optimization
- Multi-objective: PSL + Bandwidth + LPI  
- Range vs Doppler resolution trade-offs
- Communications-radar waveform fusion
- Custom radar performance metrics

**Framework Benefits:**
- Add new metrics by implementing differentiable functions
- Modify loss function for different objective combinations
- Leverage automatic differentiation for complex trade-offs

## Technical Details

### Requirements
- Python 3.7+
- PyTorch 1.8+
- NumPy, Matplotlib, PyYAML
- CUDA support (optional, auto-detected)

### Project Structure
```
graf/
├── graf.py              # Main GRAF implementation
├── config.yaml          # Configuration file
├── config_loader.py     # Configuration management
├── requirements.txt     # Dependencies
├── results/             # Output directory
└── README.md           # This file
```

### Performance Notes
- GPU acceleration automatically detected and used when available
- Memory usage scales as O(N²) for ambiguity function computation
- Recommended: N≤512 for consumer GPUs

## Contributing

We welcome contributions! Areas of interest:
- Alternative optimization algorithms
- Extended multi-objective formulations  
- TensorFlow/JAX implementations
- Performance optimizations

## License & Patent Information

**Patent Status**: Patent Pending  
**Academic Use**: Free for research and educational purposes  
**Commercial Use**: Requires licensing agreement

This software implements novel methods protected by provisional patent application. Academic and research use is encouraged and freely permitted. Commercial applications require a licensing agreement.

### For Commercial Licensing
Contact the authors for commercial licensing inquiries and partnership opportunities.

## Contact

marcoantonio.bara@esade.edu

---

**GRAF**: Making radar waveform optimization differentiable, one gradient at a time. 🎯 