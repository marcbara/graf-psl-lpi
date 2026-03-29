"""
Modal runner for v2 experiments (TAES strengthening).

Usage:
    .venv/Scripts/python.exe run_modal_v2.py

Author: Marc Bara Iniesta
"""

import modal
import json
import time
from pathlib import Path

app = modal.App("radar-v2-experiments")

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "matplotlib", "seaborn", "pyyaml")
)

# Shared code embedded in functions
SHARED_CODE = open("experiments_v2_shared.py").read() if Path("experiments_v2_shared.py").exists() else ""

# We'll embed everything inline to avoid mount issues

@app.function(gpu="T4", image=gpu_image, timeout=7200)
def run_all_v2():
    """Run all v2 experiments on GPU."""
    import torch
    import torch.nn as nn
    import numpy as np
    import time as _time
    import json
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    device = "cuda"
    logging.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---- Inline core functions ----
    class DifferentiableAmbiguity(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, s):
            N = s.shape[-1]
            idx = torch.arange(N, device=s.device)
            shift_idx = (idx.unsqueeze(0) - idx.unsqueeze(1)) % N
            S_shifted = s[..., shift_idx]
            R = s.unsqueeze(-2) * S_shifted.conj()
            X = torch.fft.fft(R, dim=-1)
            chi = (X * X.conj()).real
            chi = torch.fft.fftshift(chi, dim=(-2, -1))
            chi = chi / chi.amax(dim=(-2, -1), keepdim=True)
            return chi

    def compute_psl(chi, mainlobe_width=3):
        N = chi.shape[-1]
        peak_value = chi.max()
        flat_idx = chi.argmax()
        peak_loc = (flat_idx // N, flat_idx % N)
        i_c, j_c = torch.meshgrid(torch.arange(N, device=chi.device), torch.arange(N, device=chi.device), indexing="ij")
        dist = torch.abs(i_c - peak_loc[0]) + torch.abs(j_c - peak_loc[1])
        mask = torch.ones_like(chi)
        mask[dist <= mainlobe_width] = 0
        if peak_value < 1e-10:
            return torch.tensor(1.0, device=chi.device)
        return (chi * mask).max() / peak_value

    def compute_sv(s):
        S = torch.fft.fft(s)
        power = torch.abs(S)**2
        return torch.var(power / power.sum())

    class ResidualBlock(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.block = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Linear(dim, dim), nn.LayerNorm(dim))
            self.activation = nn.GELU()
        def forward(self, x):
            return self.activation(x + self.block(x))

    class WaveformGenerator(nn.Module):
        def __init__(self, N, hidden_dim=256, num_layers=4):
            super().__init__()
            layers = [nn.Linear(2, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()]
            for _ in range(num_layers - 2):
                layers.append(ResidualBlock(hidden_dim))
            layers.append(nn.Linear(hidden_dim, N))
            self.network = nn.Sequential(*layers)
        def forward(self, params):
            return torch.exp(1j * torch.sigmoid(self.network(params)) * 2 * np.pi)

    amb_func = DifferentiableAmbiguity().to(device)
    N = 256
    mainlobe_width = 3
    all_results = {}

    # ================================================================
    # EXPERIMENT 1: Baseline comparison (analytical 1D vs AD 2D)
    # ================================================================
    logging.info("=" * 50)
    logging.info("EXP 1: Baseline comparison")
    logging.info("=" * 50)

    lambda_values = [0.0, 0.5, 1.0, 2.0]
    num_seeds = 10
    baseline = {'analytical_1d': {'psl': [], 'time': []}, 'ad_2d': {'psl': [], 'time': []}}

    for seed in range(num_seeds):
        logging.info(f"  Seed {seed+1}/{num_seeds}")

        for method in ['analytical_1d', 'ad_2d']:
            seed_psl = []
            t0 = _time.time()

            for lam in lambda_values:
                torch.manual_seed(seed * 100 + int(lam * 100))
                phases = torch.rand(N, device=device) * 2 * np.pi
                phases.requires_grad_(True)
                opt = torch.optim.Adam([phases], lr=0.01)

                for it in range(2000):
                    opt.zero_grad()
                    s = torch.exp(1j * phases)

                    if method == 'analytical_1d':
                        # Zero-Doppler autocorrelation PSL only
                        R = torch.zeros(N, dtype=torch.complex64, device=device)
                        for k in range(N):
                            R[k] = torch.sum(s * torch.conj(torch.roll(s, k)))
                        R_mag = (R * R.conj()).real
                        R_norm = R_mag / R_mag[0]
                        m = torch.ones(N, device=device)
                        for k in range(-mainlobe_width, mainlobe_width+1):
                            m[k % N] = 0
                        psl = (R_norm * m).max()
                    else:
                        # Full 2D ambiguity function
                        chi = amb_func(s)
                        psl = compute_psl(chi, mainlobe_width)

                    lpi = compute_sv(s)
                    loss = psl + lam * lpi * 2000
                    loss.backward()
                    opt.step()

                with torch.no_grad():
                    sf = torch.exp(1j * phases)
                    cf = amb_func(sf)
                    seed_psl.append(compute_psl(cf, mainlobe_width).item())

            baseline[method]['psl'].append(seed_psl)
            baseline[method]['time'].append(_time.time() - t0)

    baseline['lambda_values'] = lambda_values
    all_results['baseline'] = baseline
    logging.info("  Baseline complete")

    # ================================================================
    # EXPERIMENT 2: Spectral mask constraint
    # ================================================================
    logging.info("=" * 50)
    logging.info("EXP 2: Spectral mask")
    logging.info("=" * 50)

    freqs = torch.linspace(0, 1, N, device=device)
    spec_mask = ((freqs > 0.21) & (freqs < 0.29)).float()

    mask_results = {'with_mask': {'psl': [], 'lpi': [], 'mask_energy': []},
                    'without_mask': {'psl': [], 'lpi': [], 'mask_energy': []}}

    for seed in range(10):
        for use_mask in [True, False]:
            key = 'with_mask' if use_mask else 'without_mask'
            torch.manual_seed(seed)
            phases = torch.rand(N, device=device) * 2 * np.pi
            phases.requires_grad_(True)
            opt = torch.optim.Adam([phases], lr=0.01)

            for it in range(2000):
                opt.zero_grad()
                s = torch.exp(1j * phases)
                chi = amb_func(s)
                psl = compute_psl(chi, mainlobe_width)
                lpi = compute_sv(s)
                S = torch.fft.fft(s)
                pw = (S * S.conj()).real
                pw_norm = pw / pw.sum()
                mask_e = (pw_norm * spec_mask).sum()

                if use_mask:
                    loss = psl + 0.5 * lpi * 2000 + 5.0 * mask_e
                else:
                    loss = psl + 0.5 * lpi * 2000
                loss.backward()
                opt.step()

            with torch.no_grad():
                sf = torch.exp(1j * phases)
                cf = amb_func(sf)
                S_f = torch.fft.fft(sf)
                pw_f = (S_f * S_f.conj()).real
                pw_fn = pw_f / pw_f.sum()
                mask_results[key]['psl'].append(compute_psl(cf, mainlobe_width).item())
                mask_results[key]['lpi'].append(compute_sv(sf).item())
                mask_results[key]['mask_energy'].append((pw_fn * spec_mask).sum().item())

    all_results['spectral_mask'] = mask_results
    logging.info("  Spectral mask complete")

    # ================================================================
    # EXPERIMENT 3: Neural generator multi-seed
    # ================================================================
    logging.info("=" * 50)
    logging.info("EXP 3: Neural multi-seed")
    logging.info("=" * 50)

    lambda_test = [0.0, 0.25, 0.5, 1.0, 2.0]
    n_seeds_neural = 10
    neural_psl, neural_lpi, neural_times = [], [], []

    for seed in range(n_seeds_neural):
        logging.info(f"  Neural seed {seed+1}/{n_seeds_neural}")
        torch.manual_seed(seed)
        model = WaveformGenerator(N, 256, 4).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)

        t0 = _time.time()
        model.train()
        for epoch in range(200):
            opt.zero_grad()
            lams = torch.rand(32, device=device) * 3.0
            noise = torch.rand(32, device=device)
            p = torch.stack([lams, noise], dim=-1)
            wfs = model(p)
            tl = torch.tensor(0.0, device=device)
            for i in range(32):
                chi = amb_func(wfs[i])
                tl = tl + compute_psl(chi, mainlobe_width) + lams[i] * compute_sv(wfs[i]) * 2000
            (tl / 32).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        neural_times.append(_time.time() - t0)

        model.eval()
        sp, sl = [], []
        for lam in lambda_test:
            with torch.no_grad():
                s = model(torch.tensor([[lam, 0.5]], device=device))[0]
                chi = amb_func(s)
                sp.append(compute_psl(chi, mainlobe_width).item())
                sl.append(compute_sv(s).item())
        neural_psl.append(sp)
        neural_lpi.append(sl)

    psl_a = np.array(neural_psl)
    lpi_a = np.array(neural_lpi)
    all_results['neural_multi_seed'] = {
        'psl_mean': psl_a.mean(axis=0).tolist(),
        'psl_std': psl_a.std(axis=0).tolist(),
        'lpi_mean': lpi_a.mean(axis=0).tolist(),
        'lpi_std': lpi_a.std(axis=0).tolist(),
        'time_mean': float(np.mean(neural_times)),
        'time_std': float(np.std(neural_times)),
        'lambda_values': lambda_test,
        'num_seeds': n_seeds_neural,
    }
    logging.info("  Neural multi-seed complete")

    # ================================================================
    # EXPERIMENT 4: Ablations
    # ================================================================
    logging.info("=" * 50)
    logging.info("EXP 4: Ablations")
    logging.info("=" * 50)

    ablation_configs = [
        ('Small (64d,2L)', 64, 2, 200),
        ('Medium (128d,3L)', 128, 3, 200),
        ('Full (256d,4L)', 256, 4, 200),
        ('Large (512d,4L)', 512, 4, 200),
        ('Full, 50ep', 256, 4, 50),
        ('Full, 100ep', 256, 4, 100),
    ]
    ablation_results = {}

    for name, hdim, nlayers, nepochs in ablation_configs:
        logging.info(f"  Ablation: {name}")
        ab_psl = []
        for seed in range(5):
            torch.manual_seed(seed)
            model = WaveformGenerator(N, hdim, nlayers).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            model.train()
            for ep in range(nepochs):
                opt.zero_grad()
                lams = torch.rand(16, device=device) * 3.0
                noise = torch.rand(16, device=device)
                p = torch.stack([lams, noise], dim=-1)
                wfs = model(p)
                tl = torch.tensor(0.0, device=device)
                for i in range(16):
                    chi = amb_func(wfs[i])
                    tl = tl + compute_psl(chi, mainlobe_width) + lams[i] * compute_sv(wfs[i]) * 2000
                (tl / 16).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            model.eval()
            sp = []
            for lam in [0.0, 0.5, 1.0, 2.0]:
                with torch.no_grad():
                    s = model(torch.tensor([[lam, 0.5]], device=device))[0]
                    chi = amb_func(s)
                    sp.append(compute_psl(chi, mainlobe_width).item())
            ab_psl.append(sp)
        pa = np.array(ab_psl)
        params = sum(p.numel() for p in model.parameters())
        ablation_results[name] = {
            'psl_db_mean': [round(20*np.log10(pa[:, i].mean()), 1) for i in range(4)],
            'psl_db_std': [round(20*pa[:, i].std()/(pa[:, i].mean()*np.log(10)), 1) for i in range(4)],
            'params': params,
        }
        logging.info(f"    PSL: {ablation_results[name]['psl_db_mean']}")

    all_results['ablations'] = ablation_results

    logging.info("=" * 50)
    logging.info("ALL V2 EXPERIMENTS COMPLETE")
    logging.info("=" * 50)

    return all_results


@app.local_entrypoint()
def main():
    print("Running v2 experiments on Modal GPU...")
    result = run_all_v2.remote()

    # Save locally
    local_dir = Path("results")
    local_dir.mkdir(exist_ok=True)
    with open(local_dir / "v2_all_results.json", "w") as f:
        json.dump(result, f, indent=2, default=float)
    print(f"Saved: results/v2_all_results.json")

    # Print summary
    import numpy as np

    print("\n--- Baseline Comparison ---")
    for method in ['analytical_1d', 'ad_2d']:
        psl = np.array(result['baseline'][method]['psl'])
        for i, lam in enumerate(result['baseline']['lambda_values']):
            print(f"  {method} lam={lam}: PSL={20*np.log10(psl[:,i].mean()):.1f}dB")

    print("\n--- Spectral Mask ---")
    for key in ['without_mask', 'with_mask']:
        psl = np.mean(result['spectral_mask'][key]['psl'])
        me = np.mean(result['spectral_mask'][key]['mask_energy'])
        print(f"  {key}: PSL={20*np.log10(psl):.1f}dB mask_energy={me:.4f}")

    print("\n--- Neural Multi-Seed ---")
    for i, lam in enumerate(result['neural_multi_seed']['lambda_values']):
        pm = result['neural_multi_seed']['psl_mean'][i]
        ps = result['neural_multi_seed']['psl_std'][i]
        print(f"  lam={lam}: PSL={20*np.log10(pm):.1f}+/-{20*ps/(pm*np.log(10)):.1f}dB")

    print("\n--- Ablations ---")
    for name, data in result['ablations'].items():
        print(f"  {name}: PSL={data['psl_db_mean']} params={data['params']:,}")

    print("\nDone!")
