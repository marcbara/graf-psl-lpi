"""
Modal GPU Experiment Runner

Runs all experiments for the IEEE TAES paper on Modal cloud GPUs.
All code is embedded in the functions to avoid mount issues.

Usage:
    .venv/Scripts/python.exe run_modal.py

Author: Marc Bara Iniesta
"""

import modal
import json
import time
from pathlib import Path

app = modal.App("radar-waveform-experiments")

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "matplotlib", "seaborn", "pyyaml")
)

results_volume = modal.Volume.from_name("radar-results", create_if_missing=True)


# ---- Shared code that runs inside Modal containers ----

SHARED_CODE = '''
import torch
import torch.nn as nn
import numpy as np
import time as _time
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# ---- Differentiable Ambiguity Function ----
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

def compute_spectral_uniformity(s, epsilon=1e-10):
    S = torch.fft.fft(s)
    power = torch.abs(S)**2
    power = power / power.sum()
    return torch.var(power)

# ---- Neural Waveform Generator ----
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
        self.N = N
        layers = [nn.Linear(2, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()]
        for _ in range(num_layers - 2):
            layers.append(ResidualBlock(hidden_dim))
        layers.append(nn.Linear(hidden_dim, N))
        self.network = nn.Sequential(*layers)
    def forward(self, design_params):
        phases = torch.sigmoid(self.network(design_params)) * 2 * np.pi
        return torch.exp(1j * phases)
'''


# ---- Experiment 1: Multi-seed GRAF vs GA ----

@app.function(
    gpu="T4",
    image=gpu_image,
    volumes={"/results": results_volume},
    timeout=7200,
)
def run_multi_seed(N: int = 256, num_seeds: int = 20):
    exec(SHARED_CODE, globals())

    device = "cuda"
    lambda_values = [0.0, 0.25, 0.5, 1.0, 2.0]
    mainlobe_width = 3
    lpi_scale_factor = 2000.0

    logging.info(f"Multi-seed: N={N}, seeds={num_seeds}, GPU={torch.cuda.get_device_name(0)}")

    results = {"gradient": {"psl": [], "lpi": [], "time": []}, "ga": {"psl": [], "lpi": [], "time": []}}
    amb_func = DifferentiableAmbiguity().to(device)

    for seed in range(num_seeds):
        logging.info(f"Seed {seed+1}/{num_seeds}")

        # Gradient optimization
        seed_psl, seed_lpi = [], []
        t0 = _time.time()
        for lam in lambda_values:
            torch.manual_seed(seed * 100 + int(lam * 100))
            phases = torch.rand(N, device=device) * 2 * np.pi
            phases.requires_grad_(True)
            opt = torch.optim.Adam([phases], lr=0.01)
            for _ in range(2000):
                opt.zero_grad()
                s = torch.exp(1j * phases)
                chi = amb_func(s)
                psl = compute_psl(chi, mainlobe_width)
                S = torch.fft.fft(s)
                pw = torch.abs(S)**2
                lpi = torch.var(pw / pw.sum())
                loss = psl + lam * lpi * lpi_scale_factor
                loss.backward()
                opt.step()
            with torch.no_grad():
                sf = torch.exp(1j * phases)
                cf = amb_func(sf)
                seed_psl.append(compute_psl(cf, mainlobe_width).item())
                seed_lpi.append(compute_spectral_uniformity(sf).item())
        results["gradient"]["psl"].append(seed_psl)
        results["gradient"]["lpi"].append(seed_lpi)
        results["gradient"]["time"].append(_time.time() - t0)

        # GA optimization
        seed_psl_ga, seed_lpi_ga = [], []
        t0 = _time.time()
        pop_size, max_gen = 50, 300
        for lam in lambda_values:
            torch.manual_seed(seed * 100 + int(lam * 100))
            pop = torch.rand(pop_size, N, device=device) * 2 * np.pi
            elite_count = 5
            for gen in range(max_gen):
                fitness = torch.zeros(pop_size, device=device)
                for i in range(pop_size):
                    s = torch.exp(1j * pop[i])
                    chi = amb_func(s)
                    with torch.no_grad():
                        psl = compute_psl(chi, mainlobe_width)
                        S = torch.fft.fft(s)
                        pw = torch.abs(S)**2
                        lpi = torch.var(pw / pw.sum())
                        fitness[i] = -(psl + lam * lpi * lpi_scale_factor)
                new_pop = torch.zeros_like(pop)
                elite_idx = torch.topk(fitness, elite_count).indices
                new_pop[:elite_count] = pop[elite_idx]
                for i in range(elite_count, pop_size):
                    tourn = torch.randint(0, pop_size, (3,))
                    new_pop[i] = pop[tourn[torch.argmax(fitness[tourn])]]
                for i in range(elite_count, pop_size - 1, 2):
                    if torch.rand(1).item() < 0.8:
                        pt = torch.randint(1, N - 1, (1,)).item()
                        tmp = new_pop[i, pt:].clone()
                        new_pop[i, pt:] = new_pop[i+1, pt:]
                        new_pop[i+1, pt:] = tmp
                mask = torch.rand(pop_size - elite_count, N, device=device) < 0.1
                mut = (torch.rand(pop_size - elite_count, N, device=device) - 0.5) * 0.25
                new_pop[elite_count:] = torch.where(mask, (new_pop[elite_count:] + mut) % (2*np.pi), new_pop[elite_count:])
                pop = new_pop
            best = torch.argmax(fitness)
            with torch.no_grad():
                sb = torch.exp(1j * pop[best])
                cb = amb_func(sb)
                seed_psl_ga.append(compute_psl(cb, mainlobe_width).item())
                seed_lpi_ga.append(compute_spectral_uniformity(sb).item())
        results["ga"]["psl"].append(seed_psl_ga)
        results["ga"]["lpi"].append(seed_lpi_ga)
        results["ga"]["time"].append(_time.time() - t0)

    results["lambda_values"] = lambda_values
    results["N"] = N
    results["num_seeds"] = num_seeds
    results["gpu"] = torch.cuda.get_device_name(0)

    with open(f"/results/multi_seed_N{N}.json", "w") as f:
        json.dump(results, f, indent=2)
    results_volume.commit()
    return results


# ---- Experiment 2: Neural Waveform Generator ----

@app.function(
    gpu="T4",
    image=gpu_image,
    volumes={"/results": results_volume},
    timeout=3600,
)
def run_neural_generator(N: int = 256, num_epochs: int = 200):
    exec(SHARED_CODE, globals())

    device = "cuda"
    lambda_values = [0.0, 0.25, 0.5, 1.0, 2.0]
    mainlobe_width = 3
    lpi_scale_factor = 2000.0
    batch_size = 32
    lr = 1e-3

    logging.info(f"Neural gen: N={N}, epochs={num_epochs}, GPU={torch.cuda.get_device_name(0)}")

    torch.manual_seed(42)
    model = WaveformGenerator(N, 256, 4).to(device)
    amb_func = DifferentiableAmbiguity().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    param_count = sum(p.numel() for p in model.parameters())

    history = {"epoch": [], "loss": [], "psl": [], "lpi": [], "time": []}
    start = _time.time()

    for epoch in range(num_epochs):
        model.train()
        ep_loss, ep_psl, ep_lpi = [], [], []
        for step in range(10):
            optimizer.zero_grad()
            lambdas = torch.rand(batch_size, device=device) * 3.0
            noise = torch.rand(batch_size, device=device)
            params = torch.stack([lambdas, noise], dim=-1)
            waveforms = model(params)
            total_loss = torch.tensor(0.0, device=device)
            for i in range(batch_size):
                chi = amb_func(waveforms[i])
                psl = compute_psl(chi, mainlobe_width)
                lpi = compute_spectral_uniformity(waveforms[i])
                total_loss = total_loss + psl + lambdas[i] * lpi * lpi_scale_factor
                ep_psl.append(psl.item())
                ep_lpi.append(lpi.item())
            total_loss = total_loss / batch_size
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss.append(total_loss.item())
        scheduler.step()
        history["epoch"].append(epoch)
        history["loss"].append(float(np.mean(ep_loss)))
        history["psl"].append(float(np.mean(ep_psl)))
        history["lpi"].append(float(np.mean(ep_lpi)))
        history["time"].append(_time.time() - start)
        if epoch % 20 == 0 or epoch == num_epochs - 1:
            logging.info(f"Epoch {epoch}: loss={history['loss'][-1]:.4f} psl={history['psl'][-1]:.4f} lpi={history['lpi'][-1]:.2e}")

    training_time = _time.time() - start

    # Evaluate
    model.eval()
    eval_results = {"lambda": [], "psl": [], "lpi": [], "inference_time_ms": []}
    for lam in lambda_values:
        t0 = _time.time()
        with torch.no_grad():
            p = torch.tensor([[lam, 0.5]], device=device)
            s = model(p)[0]
        inf_ms = (_time.time() - t0) * 1000
        with torch.no_grad():
            chi = amb_func(s)
            psl = compute_psl(chi, mainlobe_width).item()
            lpi = compute_spectral_uniformity(s).item()
        eval_results["lambda"].append(lam)
        eval_results["psl"].append(psl)
        eval_results["lpi"].append(lpi)
        eval_results["inference_time_ms"].append(inf_ms)
        logging.info(f"  lambda={lam}: PSL={20*np.log10(psl):.1f}dB, LPI={lpi:.2e}, inf={inf_ms:.1f}ms")

    output = {
        "training": {"history": history, "total_time": training_time, "param_count": param_count},
        "evaluation": eval_results,
        "N": N, "gpu": torch.cuda.get_device_name(0),
    }

    with open(f"/results/neural_generator_N{N}.json", "w") as f:
        json.dump(output, f, indent=2)
    results_volume.commit()
    return output


# ---- Experiment 3: GPU vs CPU Benchmark ----

@app.function(
    gpu="T4",
    image=gpu_image,
    volumes={"/results": results_volume},
    timeout=1800,
)
def run_gpu_benchmark():
    exec(SHARED_CODE, globals())

    N_values = [64, 128, 256, 512, 1024]
    num_iter = 200

    logging.info(f"GPU benchmark: GPU={torch.cuda.get_device_name(0)}")

    results = {"N_values": N_values, "cpu": [], "gpu": [], "speedup": []}
    amb_cpu = DifferentiableAmbiguity()
    amb_gpu = DifferentiableAmbiguity().cuda()

    for N in N_values:
        s_cpu = torch.exp(1j * torch.rand(N) * 2 * np.pi)
        for _ in range(10): amb_cpu(s_cpu)
        t0 = _time.time()
        for _ in range(num_iter): amb_cpu(s_cpu)
        cpu_ms = ((_time.time() - t0) / num_iter) * 1000

        s_gpu = torch.exp(1j * torch.rand(N, device="cuda") * 2 * np.pi)
        for _ in range(20): amb_gpu(s_gpu)
        torch.cuda.synchronize()
        t0 = _time.time()
        for _ in range(num_iter): amb_gpu(s_gpu)
        torch.cuda.synchronize()
        gpu_ms = ((_time.time() - t0) / num_iter) * 1000

        sp = cpu_ms / gpu_ms
        results["cpu"].append(round(cpu_ms, 3))
        results["gpu"].append(round(gpu_ms, 3))
        results["speedup"].append(round(sp, 1))
        logging.info(f"  N={N}: CPU={cpu_ms:.3f}ms GPU={gpu_ms:.3f}ms {sp:.1f}x")

    results["gpu_name"] = torch.cuda.get_device_name(0)
    with open("/results/gpu_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    results_volume.commit()
    return results


# ---- Experiment 4: Scaling ----

@app.function(
    gpu="T4",
    image=gpu_image,
    volumes={"/results": results_volume},
    timeout=7200,
)
def run_scaling(num_seeds: int = 5):
    exec(SHARED_CODE, globals())

    device = "cuda"
    N_values = [64, 128, 256, 512]
    lam = 0.5
    mainlobe_width = 3
    lpi_scale_factor = 2000.0

    logging.info(f"Scaling: N={N_values}, seeds={num_seeds}")

    results = {}
    amb_func = DifferentiableAmbiguity().to(device)

    for N in N_values:
        max_iter = min(2000, max(200, N * 4))
        times, psls = [], []
        for seed in range(num_seeds):
            torch.manual_seed(seed)
            phases = torch.rand(N, device=device) * 2 * np.pi
            phases.requires_grad_(True)
            opt = torch.optim.Adam([phases], lr=0.01)
            t0 = _time.time()
            for _ in range(max_iter):
                opt.zero_grad()
                s = torch.exp(1j * phases)
                chi = amb_func(s)
                psl = compute_psl(chi, mainlobe_width)
                S = torch.fft.fft(s)
                pw = torch.abs(S)**2
                lpi_val = torch.var(pw / pw.sum())
                loss = psl + lam * lpi_val * lpi_scale_factor
                loss.backward()
                opt.step()
            times.append(_time.time() - t0)
            with torch.no_grad():
                sf = torch.exp(1j * phases)
                cf = amb_func(sf)
                psls.append(compute_psl(cf, mainlobe_width).item())

        results[str(N)] = {
            "time_mean": round(np.mean(times), 2), "time_std": round(np.std(times), 2),
            "psl_mean": round(np.mean(psls), 6), "psl_std": round(np.std(psls), 6),
            "psl_db_mean": round(20 * np.log10(np.mean(psls)), 1), "max_iter": max_iter,
        }
        logging.info(f"  N={N}: {np.mean(times):.1f}s, PSL={20*np.log10(np.mean(psls)):.1f}dB")

    output = {"results": results, "N_values": N_values, "lambda": lam, "num_seeds": num_seeds, "gpu": torch.cuda.get_device_name(0)}
    with open("/results/scaling_benchmark.json", "w") as f:
        json.dump(output, f, indent=2)
    results_volume.commit()
    return output


# ---- Main ----

@app.local_entrypoint()
def main():
    print("=" * 60)
    print("RADAR WAVEFORM EXPERIMENTS - Modal GPU")
    print("=" * 60)

    t0 = time.time()

    print("\n[1/4] GPU Benchmark...")
    gpu_handle = run_gpu_benchmark.spawn()

    print("[2/4] Neural generator...")
    neural_handle = run_neural_generator.spawn(N=256, num_epochs=200)

    print("[3/4] Scaling...")
    scaling_handle = run_scaling.spawn(num_seeds=5)

    print("[4/4] Multi-seed (this is the long one)...")
    multi_handle = run_multi_seed.spawn(N=256, num_seeds=20)

    # Collect
    print("\n--- Waiting for results ---")

    gpu_data = gpu_handle.get()
    print(f"\nGPU Benchmark ({gpu_data['gpu_name']}):")
    for i, N in enumerate(gpu_data["N_values"]):
        print(f"  N={N}: CPU={gpu_data['cpu'][i]}ms GPU={gpu_data['gpu'][i]}ms {gpu_data['speedup'][i]}x")

    neural_data = neural_handle.get()
    print(f"\nNeural Generator (N={neural_data['N']}):")
    print(f"  Training: {neural_data['training']['total_time']:.1f}s, params: {neural_data['training']['param_count']:,}")
    for i, lam in enumerate(neural_data["evaluation"]["lambda"]):
        import numpy as np
        pdb = 20 * np.log10(neural_data["evaluation"]["psl"][i])
        print(f"  λ={lam}: PSL={pdb:.1f}dB LPI={neural_data['evaluation']['lpi'][i]:.2e}")

    scaling_data = scaling_handle.get()
    print(f"\nScaling:")
    for N in scaling_data["N_values"]:
        r = scaling_data["results"][str(N)]
        print(f"  N={N}: {r['time_mean']}±{r['time_std']}s PSL={r['psl_db_mean']}dB")

    multi_data = multi_handle.get()
    import numpy as np
    print(f"\nMulti-seed (N={multi_data['N']}, {multi_data['num_seeds']} seeds):")
    grad_psl = np.array(multi_data["gradient"]["psl"])
    ga_psl = np.array(multi_data["ga"]["psl"])
    grad_time = np.array(multi_data["gradient"]["time"])
    ga_time = np.array(multi_data["ga"]["time"])
    for i, lam in enumerate(multi_data["lambda_values"]):
        g = 20 * np.log10(grad_psl[:, i].mean())
        a = 20 * np.log10(ga_psl[:, i].mean())
        print(f"  λ={lam}: Grad={g:.1f}dB GA={a:.1f}dB diff={g-a:.1f}dB")
    print(f"  Time: Grad={grad_time.mean():.1f}±{grad_time.std():.1f}s  GA={ga_time.mean():.1f}±{ga_time.std():.1f}s")

    # Save locally
    local_dir = Path("results")
    local_dir.mkdir(exist_ok=True)
    for name, data in [("gpu_benchmark.json", gpu_data), ("multi_seed_N256.json", multi_data),
                        ("neural_generator_N256.json", neural_data), ("scaling_benchmark.json", scaling_data)]:
        with open(local_dir / name, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Saved: results/{name}")

    print(f"\nAll done in {time.time()-t0:.0f}s")
