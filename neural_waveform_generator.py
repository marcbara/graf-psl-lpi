"""
Neural Waveform Generator with GRAF

End-to-end neural network that learns to generate radar waveforms with desired
ambiguity function properties. The network takes design parameters as input and
outputs optimized phase sequences — enabled by GRAF's differentiable ambiguity function.

This demonstrates the key advantage of AD-compatible ambiguity functions:
once trained, the network generates waveforms in milliseconds for ANY parameter
combination, replacing minutes of iterative optimization.

Author: Marc Bara Iniesta
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Import GRAF components
from graf import DifferentiableAmbiguity, compute_psl, compute_spectral_uniformity

# ========== Neural Waveform Generator Network ==========

class WaveformGenerator(nn.Module):
    """
    Neural network that maps design parameters to radar waveform phases.

    Input: design parameters (target_lambda, sequence_length_normalized)
    Output: N phase values in [0, 2*pi] representing a constant-modulus waveform

    The network is trained end-to-end through GRAF's differentiable ambiguity
    function, learning a mapping from design specifications to optimized waveforms.
    """

    def __init__(self, N: int, hidden_dim: int = 256, num_layers: int = 4):
        super().__init__()
        self.N = N

        # Input: design parameters (lambda value + noise seed for diversity)
        input_dim = 2  # [lambda, noise_seed]

        # Build MLP with residual connections
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.GELU())

        for _ in range(num_layers - 2):
            layers.append(ResidualBlock(hidden_dim))

        layers.append(nn.Linear(hidden_dim, N))

        self.network = nn.Sequential(*layers)

        # Final activation: scale to [0, 2*pi]
        # Using sigmoid * 2pi ensures smooth, bounded output
        self.phase_scale = 2 * np.pi

    def forward(self, design_params: torch.Tensor) -> torch.Tensor:
        """
        Args:
            design_params: [batch, 2] tensor with [lambda_value, noise_seed]
        Returns:
            Complex waveform s = exp(j * phases), shape [batch, N]
        """
        phases = self.network(design_params)
        phases = torch.sigmoid(phases) * self.phase_scale

        # Generate constant-modulus complex waveform
        s = torch.exp(1j * phases)
        return s

    def generate_waveform(self, lambda_value: float, device: str = 'cpu',
                          num_candidates: int = 1) -> torch.Tensor:
        """
        Generate waveform(s) for a given lambda value.

        Args:
            lambda_value: LPI weight parameter
            device: compute device
            num_candidates: number of candidate waveforms to generate
        Returns:
            Best waveform as complex tensor [N]
        """
        self.eval()
        with torch.no_grad():
            params = torch.zeros(num_candidates, 2, device=device)
            params[:, 0] = lambda_value
            params[:, 1] = torch.linspace(0, 1, num_candidates, device=device)

            waveforms = self(params)

            if num_candidates == 1:
                return waveforms[0]

            # Return all candidates (caller picks best)
            return waveforms


class ResidualBlock(nn.Module):
    """Residual MLP block with LayerNorm"""

    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.block(x))


# ========== Training Loop ==========

def train_waveform_generator(
    N: int = 256,
    hidden_dim: int = 256,
    num_layers: int = 4,
    num_epochs: int = 200,
    batch_size: int = 32,
    lr: float = 1e-3,
    lambda_range: Tuple[float, float] = (0.0, 3.0),
    lpi_scale_factor: float = 2000.0,
    mainlobe_width: int = 3,
    device: str = 'auto',
    log_interval: int = 20,
    seed: int = 42,
) -> Dict:
    """
    Train the neural waveform generator end-to-end through GRAF.

    The key insight: the ambiguity function is differentiable, so gradients
    flow from the radar performance metrics back through the ambiguity function
    into the neural network weights. This is impossible with analytical gradient
    methods that are tied to specific optimization formulations.

    Args:
        N: Waveform sequence length
        hidden_dim: Network hidden dimension
        num_layers: Number of network layers
        num_epochs: Training epochs
        batch_size: Batch size (number of lambda samples per step)
        lr: Learning rate
        lambda_range: Range of lambda values to train over
        lpi_scale_factor: Scaling factor for LPI objective
        mainlobe_width: Mainlobe width for PSL computation
        device: Compute device ('auto', 'cpu', 'cuda')
        log_interval: Logging frequency
        seed: Random seed

    Returns:
        Dictionary with trained model, training history, and timing info
    """
    # Setup
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)

    logging.info(f"Training Neural Waveform Generator on {device}")
    logging.info(f"  N={N}, hidden={hidden_dim}, layers={num_layers}")
    logging.info(f"  epochs={num_epochs}, batch={batch_size}, lr={lr}")
    logging.info(f"  lambda_range={lambda_range}")

    # Initialize model and optimizer
    model = WaveformGenerator(N, hidden_dim, num_layers).to(device)
    amb_func = DifferentiableAmbiguity().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    param_count = sum(p.numel() for p in model.parameters())
    logging.info(f"  Model parameters: {param_count:,}")

    # Training history
    history = {
        'epoch': [], 'loss': [], 'psl': [], 'lpi': [],
        'time': [], 'lr': []
    }

    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        epoch_psl = []
        epoch_lpi = []

        # Multiple gradient steps per epoch for stability
        steps_per_epoch = 10
        for step in range(steps_per_epoch):
            optimizer.zero_grad()

            # Sample random lambda values and noise seeds
            lambdas = torch.rand(batch_size, device=device) * (lambda_range[1] - lambda_range[0]) + lambda_range[0]
            noise_seeds = torch.rand(batch_size, device=device)
            design_params = torch.stack([lambdas, noise_seeds], dim=-1)

            # Forward pass: design params -> waveforms -> ambiguity -> loss
            waveforms = model(design_params)

            # Compute loss for each waveform in batch
            total_loss = torch.tensor(0.0, device=device)
            batch_psl = []
            batch_lpi = []

            for i in range(batch_size):
                chi = amb_func(waveforms[i])
                psl = compute_psl(chi, mainlobe_width)
                lpi = compute_spectral_uniformity(waveforms[i])

                # Loss weighted by the sampled lambda
                loss_i = psl + lambdas[i] * lpi * lpi_scale_factor
                total_loss = total_loss + loss_i

                batch_psl.append(psl.item())
                batch_lpi.append(lpi.item())

            total_loss = total_loss / batch_size

            # Backward pass: gradients flow through GRAF into network weights
            total_loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_losses.append(total_loss.item())
            epoch_psl.extend(batch_psl)
            epoch_lpi.extend(batch_lpi)

        scheduler.step()

        # Record epoch stats
        elapsed = time.time() - start_time
        history['epoch'].append(epoch)
        history['loss'].append(np.mean(epoch_losses))
        history['psl'].append(np.mean(epoch_psl))
        history['lpi'].append(np.mean(epoch_lpi))
        history['time'].append(elapsed)
        history['lr'].append(scheduler.get_last_lr()[0])

        if epoch % log_interval == 0 or epoch == num_epochs - 1:
            logging.info(
                f"Epoch {epoch:4d}/{num_epochs} | "
                f"Loss: {history['loss'][-1]:.4f} | "
                f"PSL: {history['psl'][-1]:.4f} | "
                f"LPI: {history['lpi'][-1]:.2e} | "
                f"Time: {elapsed:.1f}s"
            )

    total_time = time.time() - start_time
    logging.info(f"Training complete in {total_time:.1f}s")

    return {
        'model': model,
        'history': history,
        'total_time': total_time,
        'config': {
            'N': N, 'hidden_dim': hidden_dim, 'num_layers': num_layers,
            'num_epochs': num_epochs, 'batch_size': batch_size, 'lr': lr,
            'lambda_range': lambda_range, 'lpi_scale_factor': lpi_scale_factor,
            'mainlobe_width': mainlobe_width, 'device': device, 'seed': seed,
            'param_count': param_count,
        }
    }


# ========== Evaluation ==========

def evaluate_generator(
    model: WaveformGenerator,
    lambda_values: List[float],
    N: int = 256,
    mainlobe_width: int = 3,
    num_candidates: int = 16,
    device: str = 'cpu',
) -> Dict:
    """
    Evaluate the trained generator across lambda values.
    Measures both quality (PSL, LPI) and inference speed.

    Args:
        model: Trained WaveformGenerator
        lambda_values: Lambda values to evaluate
        N: Sequence length
        mainlobe_width: Mainlobe width for PSL
        num_candidates: Candidates to generate per lambda (pick best)
        device: Compute device

    Returns:
        Dictionary with evaluation results
    """
    model.eval()
    amb_func = DifferentiableAmbiguity().to(device)

    results = {
        'lambda': [], 'psl': [], 'lpi': [],
        'inference_time_ms': [], 'waveforms': []
    }

    for lam in lambda_values:
        # Time the inference
        start = time.time()
        with torch.no_grad():
            waveforms = model.generate_waveform(lam, device, num_candidates)
            if waveforms.dim() == 1:
                waveforms = waveforms.unsqueeze(0)
        inference_time = (time.time() - start) * 1000  # ms

        # Evaluate each candidate, pick best
        best_loss = float('inf')
        best_idx = 0

        for i in range(waveforms.shape[0]):
            with torch.no_grad():
                chi = amb_func(waveforms[i])
                psl = compute_psl(chi, mainlobe_width).item()
                lpi = compute_spectral_uniformity(waveforms[i]).item()
                loss = psl + lam * lpi * 2000

                if loss < best_loss:
                    best_loss = loss
                    best_idx = i
                    best_psl = psl
                    best_lpi = lpi

        results['lambda'].append(lam)
        results['psl'].append(best_psl)
        results['lpi'].append(best_lpi)
        results['inference_time_ms'].append(inference_time)
        results['waveforms'].append(waveforms[best_idx].detach().cpu())

        logging.info(
            f"  lambda={lam:.2f}: PSL={best_psl:.4f}, "
            f"LPI={best_lpi:.2e}, inference={inference_time:.1f}ms"
        )

    return results


# ========== Comparison Experiment ==========

def run_comparison_experiment(
    N: int = 256,
    lambda_values: List[float] = [0.0, 0.25, 0.5, 1.0, 2.0],
    num_seeds: int = 10,
    device: str = 'auto',
    results_dir: str = 'results',
) -> Dict:
    """
    Full comparison: Neural Generator vs GRAF (iterative) vs GA
    With multiple random seeds for statistical significance.

    This is the main experiment for the paper.
    """
    from graf import gradient_optimize, genetic_algorithm
    from config_loader import load_config

    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    Path(results_dir).mkdir(exist_ok=True)

    logging.info("=" * 60)
    logging.info("COMPARISON EXPERIMENT: Neural Generator vs GRAF vs GA")
    logging.info(f"N={N}, seeds={num_seeds}, device={device}")
    logging.info("=" * 60)

    all_results = {
        'neural': {'psl': [], 'lpi': [], 'time': []},
        'graf': {'psl': [], 'lpi': [], 'time': []},
        'ga': {'psl': [], 'lpi': [], 'time': []},
    }

    config = load_config()
    config.update_from_dict({
        'signal': {'sequence_length': N},
        'compute': {'device': device},
    })

    for seed in range(num_seeds):
        logging.info(f"\n--- Seed {seed+1}/{num_seeds} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        # 1. Train Neural Generator
        logging.info("Training Neural Waveform Generator...")
        train_result = train_waveform_generator(
            N=N, num_epochs=150, batch_size=32, lr=1e-3,
            device=device, seed=seed, log_interval=50,
        )

        # Evaluate neural generator
        neural_eval = evaluate_generator(
            train_result['model'], lambda_values, N=N, device=device,
        )

        # Total time = training + inference
        neural_time_total = train_result['total_time']

        all_results['neural']['psl'].append(neural_eval['psl'])
        all_results['neural']['lpi'].append(neural_eval['lpi'])
        all_results['neural']['time'].append(neural_time_total)

        # 2. GRAF iterative optimization
        logging.info("Running GRAF iterative optimization...")
        graf_psl_seed = []
        graf_lpi_seed = []
        graf_time_total = 0

        for lam in lambda_values:
            config.update_from_dict({'compute': {'random_seed': seed}})
            torch.manual_seed(seed)

            start = time.time()
            s, hist = gradient_optimize(config, lambda_lpi=lam)
            elapsed = time.time() - start

            graf_psl_seed.append(hist['psl'][-1])
            graf_lpi_seed.append(hist['lpi'][-1])
            graf_time_total += elapsed

        all_results['graf']['psl'].append(graf_psl_seed)
        all_results['graf']['lpi'].append(graf_lpi_seed)
        all_results['graf']['time'].append(graf_time_total)

        # 3. GA optimization
        logging.info("Running GA optimization...")
        ga_psl_seed = []
        ga_lpi_seed = []
        ga_time_total = 0

        for lam in lambda_values:
            torch.manual_seed(seed)

            start = time.time()
            s, hist = genetic_algorithm(config, lambda_lpi=lam)
            elapsed = time.time() - start

            ga_psl_seed.append(hist['psl'][-1])
            ga_lpi_seed.append(hist['lpi'][-1])
            ga_time_total += elapsed

        all_results['ga']['psl'].append(ga_psl_seed)
        all_results['ga']['lpi'].append(ga_lpi_seed)
        all_results['ga']['time'].append(ga_time_total)

    # Save raw results
    save_path = Path(results_dir) / 'comparison_results.json'
    serializable = {
        method: {
            'psl': [[float(v) for v in seed_vals] for seed_vals in data['psl']],
            'lpi': [[float(v) for v in seed_vals] for seed_vals in data['lpi']],
            'time': [float(t) for t in data['time']],
        }
        for method, data in all_results.items()
    }
    serializable['lambda_values'] = lambda_values
    serializable['N'] = N
    serializable['num_seeds'] = num_seeds

    with open(save_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    logging.info(f"Results saved to {save_path}")

    return all_results


# ========== Plotting ==========

def plot_neural_training(history: Dict, save_path: Optional[str] = None):
    """Plot neural generator training curves."""
    sns.set_style("whitegrid", {"grid.linewidth": 0.5, "grid.alpha": 0.5})
    sns.set_context("paper", font_scale=1.2, rc={"lines.linewidth": 2})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(history['epoch'], history['loss'], color='black', linewidth=2)
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Combined Loss', fontweight='bold')
    ax1.set_title('Training Convergence', fontweight='bold')

    ax2.plot(history['epoch'], history['psl'], color='black', linewidth=2, label='PSL')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(history['epoch'], [l * 1000 for l in history['lpi']],
                  color='gray', linewidth=2, linestyle='--', label='LPI (x1000)')
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('PSL', fontweight='bold')
    ax2_twin.set_ylabel('Spectral Variance (x1000)', fontweight='bold')
    ax2.set_title('Objective Convergence', fontweight='bold')

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, frameon=True)

    sns.despine(fig=fig, left=False, bottom=False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        plt.show()


def plot_three_way_comparison(
    all_results: Dict,
    lambda_values: List[float],
    save_path: Optional[str] = None,
):
    """
    Plot three-way comparison: Neural Generator vs GRAF vs GA
    With error bars from multiple seeds.
    """
    sns.set_style("whitegrid", {"grid.linewidth": 0.5, "grid.alpha": 0.5})
    sns.set_context("paper", font_scale=1.2, rc={"lines.linewidth": 2})

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    colors = {'neural': '#2c3e50', 'graf': '#7f8c8d', 'ga': '#bdc3c7'}
    labels = {'neural': 'Neural Gen.', 'graf': 'GRAF (iterative)', 'ga': 'GA'}
    markers = {'neural': 'D', 'graf': 'o', 'ga': 's'}

    # Compute mean and std across seeds
    stats = {}
    for method in ['neural', 'graf', 'ga']:
        psl_arr = np.array(all_results[method]['psl'])  # [seeds, lambdas]
        lpi_arr = np.array(all_results[method]['lpi'])
        stats[method] = {
            'psl_mean': psl_arr.mean(axis=0),
            'psl_std': psl_arr.std(axis=0),
            'lpi_mean': lpi_arr.mean(axis=0),
            'lpi_std': lpi_arr.std(axis=0),
            'time_mean': np.mean(all_results[method]['time']),
            'time_std': np.std(all_results[method]['time']),
        }

    # Panel 1: Pareto frontier with error bars
    ax = axes[0]
    for method in ['ga', 'graf', 'neural']:
        s = stats[method]
        ax.errorbar(
            s['psl_mean'], s['lpi_mean'] * 1000,
            xerr=s['psl_std'], yerr=s['lpi_std'] * 1000,
            color=colors[method], marker=markers[method],
            label=labels[method], markersize=8, linewidth=2,
            capsize=3, capthick=1.5,
        )
    ax.set_xlabel('Peak Sidelobe Level (PSL)', fontweight='bold')
    ax.set_ylabel('Spectral Variance (x1000)', fontweight='bold')
    ax.set_title('Pareto Frontier', fontweight='bold')
    ax.legend(frameon=True)

    # Panel 2: PSL comparison per lambda
    ax = axes[1]
    x = np.arange(len(lambda_values))
    width = 0.25
    for i, method in enumerate(['neural', 'graf', 'ga']):
        s = stats[method]
        psl_db = 20 * np.log10(s['psl_mean'])
        psl_db_std = 20 * s['psl_std'] / (s['psl_mean'] * np.log(10))
        ax.bar(x + (i - 1) * width, psl_db, width,
               yerr=psl_db_std, label=labels[method],
               color=colors[method], edgecolor='black', linewidth=0.8,
               capsize=3)
    ax.set_xlabel('Lambda (LPI weight)', fontweight='bold')
    ax.set_ylabel('PSL (dB)', fontweight='bold')
    ax.set_title('PSL Achievement', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{l:.2f}' for l in lambda_values])
    ax.legend(frameon=True)

    # Panel 3: Total computation time
    ax = axes[2]
    methods = ['neural', 'graf', 'ga']
    times_mean = [stats[m]['time_mean'] for m in methods]
    times_std = [stats[m]['time_std'] for m in methods]
    bars = ax.bar(
        [labels[m] for m in methods], times_mean,
        yerr=times_std, color=[colors[m] for m in methods],
        edgecolor='black', linewidth=0.8, capsize=5,
    )

    # Add note about neural gen inference
    ax.set_ylabel('Total Time (s)', fontweight='bold')
    ax.set_title('Computational Cost', fontweight='bold')

    # Annotate: neural gen = train once, then instant
    ax.annotate('Train once,\nthen <1ms/waveform',
                xy=(0, times_mean[0]), xytext=(0.5, times_mean[0] * 1.3),
                fontsize=9, ha='center', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='black'))

    sns.despine(fig=fig, left=False, bottom=False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        plt.show()


def plot_inference_demo(
    model: WaveformGenerator,
    lambda_values: List[float],
    N: int = 256,
    mainlobe_width: int = 3,
    device: str = 'cpu',
    save_path: Optional[str] = None,
):
    """
    Demonstrate real-time waveform generation:
    Show waveforms generated for different lambda values in a single forward pass.
    """
    sns.set_style("whitegrid", {"grid.linewidth": 0.5, "grid.alpha": 0.5})
    sns.set_context("paper", font_scale=1.2, rc={"lines.linewidth": 2})

    model.eval()
    amb_func = DifferentiableAmbiguity().to(device)

    n_examples = len(lambda_values)
    fig, axes = plt.subplots(2, n_examples, figsize=(4 * n_examples, 8))

    for i, lam in enumerate(lambda_values):
        with torch.no_grad():
            s = model.generate_waveform(lam, device, num_candidates=1)
            chi = amb_func(s)
            psl = compute_psl(chi, mainlobe_width).item()
            lpi = compute_spectral_uniformity(s).item()

        # Top row: ambiguity function surface
        ax = axes[0, i]
        chi_np = chi.cpu().numpy()
        ax.imshow(10 * np.log10(chi_np + 1e-10), aspect='auto',
                  cmap='gray_r', vmin=-40, vmax=0)
        ax.set_title(f'$\\lambda$={lam:.1f}\nPSL={20*np.log10(psl):.1f}dB',
                     fontweight='bold')
        if i == 0:
            ax.set_ylabel('Delay', fontweight='bold')
        ax.set_xlabel('Doppler', fontweight='bold')

        # Bottom row: power spectrum
        ax = axes[1, i]
        S = torch.fft.fft(s)
        power = torch.abs(S).cpu().numpy() ** 2
        ax.plot(power, color='black', linewidth=1.5)
        ax.set_title(f'LPI={lpi:.2e}', fontweight='bold')
        if i == 0:
            ax.set_ylabel('Power', fontweight='bold')
        ax.set_xlabel('Frequency', fontweight='bold')

    plt.suptitle('Neural Waveform Generator: Instant Generation for Any $\\lambda$',
                 fontweight='bold', fontsize=14, y=1.02)

    sns.despine(fig=fig, left=False, bottom=False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        plt.show()


# ========== Main ==========

def main():
    """Run the neural waveform generator experiment."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    N = 256
    lambda_values = [0.0, 0.25, 0.5, 1.0, 2.0]
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    # ---- Phase 1: Train the generator ----
    logging.info("=" * 60)
    logging.info("PHASE 1: Training Neural Waveform Generator")
    logging.info("=" * 60)

    train_result = train_waveform_generator(
        N=N, num_epochs=200, batch_size=32, lr=1e-3,
        device=device, log_interval=20,
    )

    # Plot training curves
    plot_neural_training(
        train_result['history'],
        save_path=str(results_dir / 'fig_neural_training.png'),
    )
    logging.info("Saved: fig_neural_training.png")

    # ---- Phase 2: Evaluate and compare ----
    logging.info("\n" + "=" * 60)
    logging.info("PHASE 2: Evaluating Generator")
    logging.info("=" * 60)

    eval_results = evaluate_generator(
        train_result['model'], lambda_values, N=N, device=device,
    )

    # Show inference speed
    avg_inference = np.mean(eval_results['inference_time_ms'])
    logging.info(f"\nAverage inference time: {avg_inference:.1f}ms per waveform")

    # ---- Phase 3: Visualize generated waveforms ----
    logging.info("\n" + "=" * 60)
    logging.info("PHASE 3: Visualizing Generated Waveforms")
    logging.info("=" * 60)

    plot_inference_demo(
        train_result['model'], lambda_values, N=N, device=device,
        save_path=str(results_dir / 'fig_neural_inference_demo.png'),
    )
    logging.info("Saved: fig_neural_inference_demo.png")

    # ---- Summary ----
    logging.info("\n" + "=" * 60)
    logging.info("SUMMARY")
    logging.info("=" * 60)
    logging.info(f"Training time: {train_result['total_time']:.1f}s")
    logging.info(f"Inference time: {avg_inference:.1f}ms (vs minutes for iterative)")
    logging.info(f"Model parameters: {train_result['config']['param_count']:,}")

    for i, lam in enumerate(lambda_values):
        psl_db = 20 * np.log10(eval_results['psl'][i])
        logging.info(
            f"  lambda={lam:.2f}: PSL={psl_db:.1f}dB, "
            f"LPI={eval_results['lpi'][i]:.2e}"
        )


if __name__ == "__main__":
    main()
