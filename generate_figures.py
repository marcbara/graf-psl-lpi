"""
Generate all figures for the IEEE TAES paper.

All two-panel figures use vertical (2,1) layout for IEEE single-column readability.

Author: Marc Bara Iniesta
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from pathlib import Path

# Publication style
sns.set_style("whitegrid", {"grid.linewidth": 0.5, "grid.alpha": 0.3})
sns.set_context("paper", font_scale=1.0, rc={"lines.linewidth": 1.0})
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'figure.dpi': 300,
})

COL_W = 3.45  # IEEE single column width in inches

RESULTS_DIR = Path("results")
FIGS_DIR = Path("IEEE_TAES/figures")
FIGS_DIR.mkdir(exist_ok=True)

C_GRAD = '#1a1a1a'
C_GA = '#999999'
C_NEURAL = '#444444'


def load_json(name):
    with open(RESULTS_DIR / name) as f:
        return json.load(f)


def save_fig(name):
    plt.savefig(FIGS_DIR / f"{name}.pdf", bbox_inches='tight')
    plt.savefig(FIGS_DIR / f"{name}.png", bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  Saved: {name}")


# ============================================================
# Fig 1: Pareto frontier (single panel — no change needed)
# ============================================================
def fig_pareto():
    d = load_json("multi_seed_N256.json")
    gp = np.array(d["gradient"]["psl"])
    ap = np.array(d["ga"]["psl"])
    gl = np.array(d["gradient"]["lpi"])
    al = np.array(d["ga"]["lpi"])
    lambdas = d["lambda_values"]

    fig, ax = plt.subplots(figsize=(COL_W, 2.8))

    ax.errorbar(ap.mean(0), al.mean(0)*1e6, xerr=ap.std(0), yerr=al.std(0)*1e6,
                color=C_GA, marker='s', label='GA', markersize=6, linewidth=0.9,
                capsize=3, markerfacecolor='white', markeredgecolor=C_GA, markeredgewidth=1.3)
    ax.errorbar(gp.mean(0), gl.mean(0)*1e6, xerr=gp.std(0), yerr=gl.std(0)*1e6,
                color=C_GRAD, marker='o', label='Gradient (ours)', markersize=6,
                linewidth=0.9, capsize=3)

    for i, lam in enumerate(lambdas):
        if i % 2 == 0:
            ax.annotate(f'$\\lambda$={lam}', (gp.mean(0)[i], gl.mean(0)[i]*1e6),
                        xytext=(8, 5), textcoords='offset points', fontsize=6.5, alpha=0.7)

    ax.set_xlabel('Peak Sidelobe Level (PSL)')
    ax.set_ylabel('Spectral Variance ($\\times 10^{-6}$)')
    ax.set_title('Pareto Frontier (mean $\\pm$ std, $n$=20)')
    ax.legend(frameon=True, fontsize=7.5)
    sns.despine(ax=ax)
    plt.tight_layout()
    save_fig("fig_pareto")


# ============================================================
# Fig 3: PSL + Time comparison — VERTICAL
# ============================================================
def fig_psl_comparison():
    d = load_json("multi_seed_N256.json")
    gp = np.array(d["gradient"]["psl"])
    ap = np.array(d["ga"]["psl"])
    gt = np.array(d["gradient"]["time"])
    at = np.array(d["ga"]["time"])
    lambdas = d["lambda_values"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(COL_W, 4.8))

    x = np.arange(len(lambdas))
    w = 0.35

    g_db = 20 * np.log10(gp.mean(0))
    g_err = 20 * gp.std(0) / (gp.mean(0) * np.log(10))
    a_db = 20 * np.log10(ap.mean(0))
    a_err = 20 * ap.std(0) / (ap.mean(0) * np.log(10))

    ax1.bar(x - w/2, g_db, w, yerr=g_err, label='Gradient', color='#666666',
            edgecolor='black', linewidth=0.5, capsize=3)
    ax1.bar(x + w/2, a_db, w, yerr=a_err, label='GA', color='#cccccc',
            edgecolor='black', linewidth=0.5, capsize=3)
    for i in range(len(lambdas)):
        diff = g_db[i] - a_db[i]
        ax1.text(i, max(g_db[i], a_db[i]) + 0.3, f'{diff:.1f} dB', ha='center',
                 fontsize=6.5, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.12', facecolor='white', alpha=0.9,
                           edgecolor='gray', linewidth=0.4))
    ax1.set_xlabel('$\\lambda$ (LPI weight)')
    ax1.set_ylabel('PSL (dB)')
    ax1.set_title('(a) Peak Sidelobe Level')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{l:.2f}' for l in lambdas])
    ax1.legend(frameon=True, fontsize=7)

    speedup = at.mean() / gt.mean()
    bars = ax2.bar(['Gradient', 'GA'], [gt.mean(), at.mean()], yerr=[gt.std(), at.std()],
                   color=['#666666', '#cccccc'], edgecolor='black', linewidth=0.5, capsize=5, width=0.5)
    ax2.set_ylabel('Total Time (s)')
    ax2.set_title(f'(b) Computation Time ({speedup:.1f}$\\times$ speedup, $n$=20)')

    for ax in [ax1, ax2]:
        sns.despine(ax=ax)
    plt.tight_layout()
    save_fig("fig_psl_comparison")


# ============================================================
# Fig 4: Neural training — VERTICAL
# ============================================================
def fig_neural_training():
    d = load_json("neural_generator_N256.json")
    h = d["training"]["history"]
    epochs = h["epoch"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(COL_W, 4.8))

    ax1.plot(epochs, h["loss"], color=C_GRAD, linewidth=0.9)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Combined Loss')
    ax1.set_title('(a) Training Convergence')
    ax1.set_xlim(0, max(epochs))

    ax2.plot(epochs, h["psl"], color=C_GRAD, linewidth=0.9, label='PSL')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('PSL (linear)')
    ax2.set_title('(b) Objective Convergence')

    ax2_twin = ax2.twinx()
    ax2_twin.plot(epochs, [l*1e6 for l in h["lpi"]], color=C_GA, linewidth=0.9,
                  linestyle='--', label='SV ($\\times 10^{-6}$)')
    ax2_twin.set_ylabel('Spectral Var. ($\\times 10^{-6}$)')

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, frameon=True, fontsize=7, loc='upper right')

    for ax in [ax1, ax2]:
        sns.despine(ax=ax)
    sns.despine(ax=ax2_twin, right=False)
    plt.tight_layout()
    save_fig("fig_neural_training")


# ============================================================
# Fig 5: Generated waveforms — keep 2x4 but full width
# ============================================================
def fig_generated_waveforms():
    import torch
    import torch.nn as nn
    import sys
    sys.path.insert(0, ".")
    from neural_waveform_generator import WaveformGenerator
    from graf import DifferentiableAmbiguity, compute_psl, compute_spectral_uniformity

    N = 256
    torch.manual_seed(42)
    np.random.seed(42)

    model = WaveformGenerator(N, hidden_dim=256, num_layers=4)
    amb_func = DifferentiableAmbiguity()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    print("  Training quick model for visualization (CPU)...")
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        lambdas = torch.rand(16) * 3.0
        noise = torch.rand(16)
        params = torch.stack([lambdas, noise], dim=-1)
        waveforms = model(params)
        total_loss = torch.tensor(0.0)
        for i in range(16):
            chi = amb_func(waveforms[i])
            psl = compute_psl(chi, 3)
            lpi = compute_spectral_uniformity(waveforms[i])
            total_loss = total_loss + psl + lambdas[i] * lpi * 2000
        (total_loss / 16).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if epoch % 25 == 0:
            print(f"    Epoch {epoch}: loss={total_loss.item():.4f}")

    lambda_values = [0.0, 0.5, 1.0, 2.0]
    lambda_labels = ['$\\lambda=0$\n(PSL focus)', '$\\lambda=0.5$\n(Balanced)',
                     '$\\lambda=1.0$', '$\\lambda=2.0$\n(LPI focus)']

    model.eval()
    fig, axes = plt.subplots(2, 4, figsize=(7.16, 3.5))  # full text width

    for i, (lam, label) in enumerate(zip(lambda_values, lambda_labels)):
        with torch.no_grad():
            s = model(torch.tensor([[lam, 0.5]]))[0]
            chi = amb_func(s)
            psl = compute_psl(chi, 3).item()
            lpi = compute_spectral_uniformity(s).item()

        ax = axes[0, i]
        chi_db = 10 * np.log10(chi.numpy() + 1e-12)
        ax.imshow(chi_db, aspect='auto', cmap='gray_r', vmin=-40, vmax=0,
                  extent=[-N//2, N//2, -N//2, N//2])
        ax.set_title(f'{label}\nPSL={20*np.log10(psl):.1f} dB', fontsize=7)
        if i == 0: ax.set_ylabel('Delay', fontsize=7)
        ax.set_xlabel('Doppler', fontsize=6)
        ax.tick_params(labelsize=5)

        ax = axes[1, i]
        power = torch.abs(torch.fft.fft(s)).numpy() ** 2
        ax.plot(power, color=C_GRAD, linewidth=1)
        ax.set_title(f'SV={lpi:.2e}', fontsize=7)
        if i == 0: ax.set_ylabel('Power', fontsize=7)
        ax.set_xlabel('Frequency bin', fontsize=6)
        ax.tick_params(labelsize=5)

    plt.tight_layout()
    save_fig("fig_generated_waveforms")


# ============================================================
# Fig 8: GPU scaling — VERTICAL
# ============================================================
def fig_gpu_scaling():
    gpu = load_json("gpu_benchmark.json")
    N_gpu = gpu["N_values"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(COL_W, 4.8))

    ax1.plot(N_gpu, gpu["cpu"], 'o-', color=C_GA, label='CPU', markersize=5, linewidth=0.9)
    ax1.plot(N_gpu, gpu["gpu"], 's-', color=C_GRAD, label=f'GPU ({gpu["gpu_name"]})',
             markersize=5, linewidth=0.9)
    ax1.set_xlabel('Sequence Length $N$')
    ax1.set_ylabel('Forward Pass Time (ms)')
    ax1.set_title('(a) Ambiguity Function: CPU vs GPU')
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.legend(frameon=True, fontsize=7)

    ax2.bar([str(N) for N in N_gpu], gpu["speedup"], color=C_GRAD,
            edgecolor='black', linewidth=0.5)
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5, linewidth=0.6)
    ax2.set_xlabel('Sequence Length $N$')
    ax2.set_ylabel('GPU Speedup ($\\times$)')
    ax2.set_title('(b) GPU Acceleration Factor')
    max_idx = np.argmax(gpu["speedup"])
    ax2.text(max_idx, gpu["speedup"][max_idx] + 1.5,
             f'{gpu["speedup"][max_idx]}$\\times$', ha='center', fontweight='bold', fontsize=8)

    for ax in [ax1, ax2]:
        sns.despine(ax=ax)
    plt.tight_layout()
    save_fig("fig_gpu_scaling")


# ============================================================
# Fig 9: N scaling — VERTICAL
# ============================================================
def fig_n_scaling():
    sc = load_json("scaling_benchmark.json")
    N_values = sc["N_values"]
    times = [sc["results"][str(N)]["time_mean"] for N in N_values]
    times_std = [sc["results"][str(N)]["time_std"] for N in N_values]
    psls = [sc["results"][str(N)]["psl_db_mean"] for N in N_values]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(COL_W, 4.8))

    ax1.errorbar(N_values, times, yerr=times_std, color=C_GRAD, marker='o',
                 markersize=5, linewidth=0.9, capsize=3)
    ax1.set_xlabel('Sequence Length $N$')
    ax1.set_ylabel('Optimization Time (s)')
    ax1.set_title('(a) Gradient Optimization Time (GPU)')
    ax1.set_xscale('log', base=2)

    ax2.plot(N_values, psls, 'o-', color=C_GRAD, markersize=5, linewidth=0.9)
    ax2.set_xlabel('Sequence Length $N$')
    ax2.set_ylabel('PSL (dB)')
    ax2.set_title('(b) PSL vs Waveform Length ($\\lambda=0.5$)')
    ax2.set_xscale('log', base=2)
    ax2.annotate('$\\approx$6 dB per\ndoubling of $N$',
                 xy=(256, -34.4), xytext=(350, -28), fontsize=7, ha='center',
                 arrowprops=dict(arrowstyle='->', color='gray'))

    for ax in [ax1, ax2]:
        sns.despine(ax=ax)
    plt.tight_layout()
    save_fig("fig_n_scaling")


# ============================================================
if __name__ == "__main__":
    print("Generating figures for IEEE TAES paper...")
    print()
    fig_pareto()
    fig_psl_comparison()
    fig_neural_training()
    fig_gpu_scaling()
    fig_n_scaling()
    fig_generated_waveforms()
    print()
    print(f"All figures saved to {FIGS_DIR}/")
