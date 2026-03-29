"""
Downstream Radar Validation Experiments

1. Target detection: show waveform PSL directly impacts weak-target detectability
2. Interference rejection: spectral mask suppresses narrowband interferer

Author: Marc Bara Iniesta
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

from graf import DifferentiableAmbiguity, compute_psl, compute_spectral_uniformity

sns.set_style("whitegrid", {"grid.linewidth": 0.5, "grid.alpha": 0.3})
sns.set_context("paper", font_scale=1.1, rc={"lines.linewidth": 2})
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'figure.dpi': 300,
})

FIGS_DIR = Path("IEEE_TAES/figures")
FIGS_DIR.mkdir(exist_ok=True)


def optimize_waveform(N, lam, method='2d', device='cpu', iterations=2000, seed=0, mask=None, lam_mask=5.0):
    """Optimize waveform with specified method."""
    torch.manual_seed(seed)
    amb = DifferentiableAmbiguity().to(device)
    phases = torch.rand(N, device=device) * 2 * np.pi
    phases.requires_grad_(True)
    opt = torch.optim.Adam([phases], lr=0.01)

    for _ in range(iterations):
        opt.zero_grad()
        s = torch.exp(1j * phases)

        if method == '2d':
            chi = amb(s)
            psl = compute_psl(chi, 3)
        elif method == '1d':
            R = torch.zeros(N, dtype=torch.complex64, device=device)
            for k in range(N):
                R[k] = torch.sum(s * torch.conj(torch.roll(s, k)))
            R_mag = (R * R.conj()).real
            R_norm = R_mag / R_mag[0]
            m = torch.ones(N, device=device)
            for k in range(-3, 4):
                m[k % N] = 0
            psl = (R_norm * m).max()

        lpi = compute_spectral_uniformity(s)
        loss = psl + lam * lpi * 2000

        if mask is not None:
            S = torch.fft.fft(s)
            pw = (S * S.conj()).real
            loss = loss + lam_mask * (pw / pw.sum() * mask.to(device)).sum()

        loss.backward()
        opt.step()

    return torch.exp(1j * phases).detach()


def optimize_waveform_ga(N, lam, device='cpu', generations=300, seed=0):
    """GA optimization."""
    torch.manual_seed(seed)
    amb = DifferentiableAmbiguity().to(device)
    pop_size = 50
    pop = torch.rand(pop_size, N, device=device) * 2 * np.pi
    for gen in range(generations):
        fitness = torch.zeros(pop_size, device=device)
        for i in range(pop_size):
            s = torch.exp(1j * pop[i])
            chi = amb(s)
            with torch.no_grad():
                psl = compute_psl(chi, 3)
                lpi = compute_spectral_uniformity(s)
                fitness[i] = -(psl + lam * lpi * 2000)
        new_pop = torch.zeros_like(pop)
        elite_idx = torch.topk(fitness, 5).indices
        new_pop[:5] = pop[elite_idx]
        for i in range(5, pop_size):
            tourn = torch.randint(0, pop_size, (3,))
            new_pop[i] = pop[tourn[torch.argmax(fitness[tourn])]]
        for i in range(5, pop_size - 1, 2):
            if torch.rand(1).item() < 0.8:
                pt = torch.randint(1, N - 1, (1,)).item()
                tmp = new_pop[i, pt:].clone()
                new_pop[i, pt:] = new_pop[i + 1, pt:]
                new_pop[i + 1, pt:] = tmp
        m = torch.rand(pop_size - 5, N, device=device) < 0.1
        mut = (torch.rand(pop_size - 5, N, device=device) - 0.5) * 0.25
        new_pop[5:] = torch.where(m, (new_pop[5:] + mut) % (2 * np.pi), new_pop[5:])
        pop = new_pop
    return torch.exp(1j * pop[torch.argmax(fitness)]).detach()


# ============================================================
# 1. TARGET DETECTION EXPERIMENT
# ============================================================

def run_detection_experiment(N=256, device='cpu'):
    """
    Demonstrate operational impact of PSL improvement.

    The ambiguity function chi[k,m] IS the matched-filter response to a
    point target at delay k, Doppler m. For a two-target scene, the
    range-Doppler map is the superposition of shifted ambiguity functions.

    We show that lower PSL directly translates to:
    - Better peak-to-sidelobe ratio around the main target
    - Better detectability of a weak secondary target
    - Fewer sidelobe-induced false alarms
    """
    print("Optimizing waveforms for detection experiment...")
    lam = 0.5
    amb = DifferentiableAmbiguity()

    waveforms = {
        'GA': optimize_waveform_ga(N, lam, device, seed=42),
        '1D Autocorr.': optimize_waveform(N, lam, '1d', device, seed=42),
        'Full 2D (ours)': optimize_waveform(N, lam, '2d', device, seed=42),
    }

    # Scene: strong target + weak target 15dB below
    strong_target = (20, 30)     # (delay_idx, doppler_idx) from center
    weak_target = (40, -20)      # relative to center
    weak_amplitude = 10 ** (-15 / 20)  # -15 dB

    # Noise floor (SNR = 30 dB)
    noise_level = 10 ** (-30 / 10)

    print("Computing range-Doppler maps...")
    results = {}
    rd_maps = {}

    for name, wf in waveforms.items():
        chi = amb(wf).numpy()  # [N, N], centered, normalized to [0,1]
        center = N // 2

        # Build RD map: superposition of shifted ambiguity functions
        rd_map = np.zeros((N, N))
        # Strong target
        rd_map += np.roll(np.roll(chi, strong_target[0], axis=0), strong_target[1], axis=1)
        # Weak target
        rd_map += weak_amplitude * np.roll(np.roll(chi, weak_target[0], axis=0), weak_target[1], axis=1)

        # Add noise
        np.random.seed(42)
        rd_map += noise_level * np.abs(np.random.randn(N, N))
        rd_map = rd_map / rd_map.max()

        rd_db = 10 * np.log10(rd_map + 1e-12)
        rd_maps[name] = rd_db

        # Metrics
        strong_loc = (center + strong_target[0], center + strong_target[1])
        weak_loc = (center + weak_target[0], center + weak_target[1])

        strong_peak = rd_db[strong_loc[0], strong_loc[1]]
        weak_peak = rd_db[weak_loc[0], weak_loc[1]]

        # Max sidelobe: exclude regions around both targets
        mask = np.ones((N, N), dtype=bool)
        for loc in [strong_loc, weak_loc]:
            for di in range(-5, 6):
                for dj in range(-5, 6):
                    mask[(loc[0]+di) % N, (loc[1]+dj) % N] = False
        max_sidelobe = rd_db[mask].max()

        # False alarms: cells above threshold outside target regions
        threshold_db = -25
        false_alarms = (rd_db[mask] > threshold_db).sum()
        total_cells = mask.sum()

        psl_wf = 20 * np.log10(compute_psl(torch.tensor(chi), 3).item())

        results[name] = {
            'waveform_psl_db': psl_wf,
            'strong_peak_db': strong_peak,
            'weak_peak_db': weak_peak,
            'max_sidelobe_db': max_sidelobe,
            'peak_to_sidelobe_db': strong_peak - max_sidelobe,
            'weak_above_sidelobes_db': weak_peak - max_sidelobe,
            'false_alarm_rate': false_alarms / total_cells,
        }
        print(f"  {name:16s}: PSL={psl_wf:.1f}dB  P/SL={strong_peak-max_sidelobe:.1f}dB  "
              f"Weak-SL={weak_peak-max_sidelobe:.1f}dB  FAR={false_alarms/total_cells:.4f}")

    # ---- Plot: zero-Doppler cut (single panel) ----
    fig, ax = plt.subplots(figsize=(3.45, 2.8))
    center = N // 2

    colors = {'GA': '#aaaaaa', '1D Autocorr.': '#666666', 'Full 2D (ours)': '#000000'}
    linestyles = {'GA': '--', '1D Autocorr.': '-.', 'Full 2D (ours)': '-'}

    delays = np.arange(N) - center

    # Plot in order: 1D first (highest sidelobes, background), then GA, then ours on top
    for name in ['1D Autocorr.', 'GA', 'Full 2D (ours)']:
        chi = amb(waveforms[name]).numpy()
        psl_db = results[name]['waveform_psl_db']

        cut = 10 * np.log10(chi[:, center] + 1e-12)
        ax.plot(delays, cut, color=colors[name], linestyle=linestyles[name],
                linewidth=0.5, label=f'{name} ({psl_db:.1f} dB)', alpha=0.85)

    ax.set_ylim(-50, 5)
    ax.set_xlabel('Delay bin offset from peak')
    ax.set_ylabel('Ambiguity (dB)')
    ax.set_title('Zero-Doppler Cut')
    ax.legend(fontsize=6.5, frameon=True, loc='upper right')
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "fig_detection.pdf", bbox_inches='tight')
    plt.savefig(FIGS_DIR / "fig_detection.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("  Saved: fig_detection")

    return results


# ============================================================
# 2. INTERFERENCE REJECTION EXPERIMENT
# ============================================================

def run_interference_experiment(N=256, device='cpu'):
    """
    Show that spectral mask constraint translates to interference rejection
    in the range-Doppler map.
    """
    print("Optimizing waveforms for interference experiment...")
    lam = 0.5
    amb = DifferentiableAmbiguity()

    freqs = torch.linspace(0, 1, N, device=device)
    spec_mask = ((freqs > 0.21) & (freqs < 0.29)).float()

    wf_nomask = optimize_waveform(N, lam, '2d', device, seed=42)
    wf_masked = optimize_waveform(N, lam, '2d', device, seed=42, mask=spec_mask, lam_mask=5.0)

    # Compare spectra
    S_nomask = torch.fft.fft(wf_nomask)
    S_masked = torch.fft.fft(wf_masked)
    pw_nomask = (S_nomask * S_nomask.conj()).real.numpy()
    pw_masked = (S_masked * S_masked.conj()).real.numpy()

    # Notch energy
    notch_idx = (freqs > 0.21) & (freqs < 0.29)
    energy_nomask = pw_nomask[notch_idx.numpy()].sum() / pw_nomask.sum()
    energy_masked = pw_masked[notch_idx.numpy()].sum() / pw_masked.sum()

    # PSL comparison
    chi_nomask = amb(wf_nomask)
    chi_masked = amb(wf_masked)
    psl_nomask = 20 * np.log10(compute_psl(chi_nomask, 3).item())
    psl_masked = 20 * np.log10(compute_psl(chi_masked, 3).item())

    print(f"  No mask:   PSL={psl_nomask:.1f}dB  notch energy={energy_nomask*100:.1f}%")
    print(f"  With mask: PSL={psl_masked:.1f}dB  notch energy={energy_masked*100:.2f}%")

    results = {
        'no_mask': {'psl_db': psl_nomask, 'notch_energy_pct': energy_nomask * 100},
        'with_mask': {'psl_db': psl_masked, 'notch_energy_pct': energy_masked * 100},
    }

    # ---- Plot: spectrum only ----
    fig, ax = plt.subplots(figsize=(3.45, 2.8))

    freqs_np = freqs.numpy()

    ax.plot(freqs_np, 10*np.log10(pw_nomask/pw_nomask.max()+1e-12),
            color='gray', linewidth=0.6, label='Without mask')
    ax.plot(freqs_np, 10*np.log10(pw_masked/pw_masked.max()+1e-12),
            color='black', linewidth=0.6, label='With mask')
    ax.axvspan(0.21, 0.29, alpha=0.15, color='red', label='Forbidden band')
    ax.set_xlabel('Normalized Frequency')
    ax.set_ylabel('Power (dB)')
    ax.set_title('Waveform Power Spectra')
    ax.set_ylim(-40, 5)
    ax.legend(fontsize=7, frameon=True)
    sns.despine(ax=ax)

    plt.tight_layout()
    plt.savefig(FIGS_DIR / "fig_interference.pdf", bbox_inches='tight')
    plt.savefig(FIGS_DIR / "fig_interference.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("  Saved: fig_interference")

    return results


if __name__ == "__main__":
    print("=" * 50)
    print("DOWNSTREAM RADAR VALIDATION")
    print("=" * 50)

    det = run_detection_experiment(N=256)
    print()
    interf = run_interference_experiment(N=256)

    all_results = {
        'detection': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in det.items()},
        'interference': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in interf.items()},
    }
    with open('results/radar_validation.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved: results/radar_validation.json")
