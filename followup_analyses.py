import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import Ridge
from tasks.poisson_dms import generate_trials

CHECKPOINT_ROOT = Path("experiments/checkpoints")
DELAY_START = 40
DELAY_END = 165
SAMPLE_START = 15
SAMPLE_END = 40
DT = 20 # ms per timestep
LAMBDAS = [0, 2, 4, 6, 8, 10]
SEEDS = [0, 1, 2]
HIDDEN_DIM = 128

def hidden_states_path(lam, seed):
    if lam == 0:
        return CHECKPOINT_ROOT / f"lam0_seed{seed}" / "hidden_states_val.npy"
    return CHECKPOINT_ROOT / f"finetuned_lam{lam}_seed{seed}" / "hidden_states_val_matched.npy"

def participation_ratio(X):
    X = X - X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    eigs = np.linalg.eigvalsh(cov)
    eigs = np.clip(eigs, 0, None)
    return eigs.sum() ** 2 / (eigs ** 2).sum()


def get_trial_metadata(lam, seed, n_trials=50):
    val_seed = seed * 1000 + 99
    _, _, infos = generate_trials(num_trials=500, lam=lam, seed=val_seed)
    infos = infos[:n_trials]
    return infos

def pr_on_clean_trials(model_loader_fn):
    X_clean, _, _ = generate_trials(num_trials=50, lam=0.0, seed=9999)

    results = []
    for lam in LAMBDAS:
        for seed in SEEDS:
            model = model_loader_fn(lam, seed)

            with torch.no_grad():
                x_tensor = torch.tensor(X_clean, dtype=torch.float32)
                _, hidden_states = model(x_tensor)
                H = hidden_states.cpu().numpy() # (50, 215, 128)

            delay = H[:, DELAY_START:DELAY_END, :].reshape(-1, HIDDEN_DIM)
            pr = participation_ratio(delay)
            results.append((lam, seed, pr))
            print(f"[clean trials] lam={lam} seed={seed}  PR={pr:.3f}")

    return results


def plot_pr_clean_vs_original(clean_results, original_results, save_path="pr_clean_vs_original.png"):
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, results, color in [("Original (with distractors)", original_results, "C0"),
                                   ("Clean trials (no distractors)", clean_results, "C3")]:
        means, sems = [], []
        for lam in LAMBDAS:
            prs = [pr for l, s, pr in results if l == lam]
            means.append(np.mean(prs))
            sems.append(np.std(prs, ddof=1) / np.sqrt(len(prs)))
        ax.errorbar(LAMBDAS, means, yerr=sems, marker="o", capsize=4, label=label, color=color)

    ax.set_xlabel("λ (training distractor intensity)")
    ax.set_ylabel("Participation Ratio")
    ax.set_title("PR on clean vs distractor-present trials")
    ax.legend()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def compute_eigenvalue_spectra():

    spectra = {}
    for lam in LAMBDAS:
        for seed in SEEDS:
            H = np.load(hidden_states_path(lam, seed))
            delay = H[:, DELAY_START:DELAY_END, :].reshape(-1, HIDDEN_DIM)
            delay = delay - delay.mean(axis=0)

            cov = np.cov(delay, rowvar=False)
            eigs = np.linalg.eigvalsh(cov)
            eigs = np.clip(eigs, 0, None)
            eigs = np.sort(eigs)[::-1] 

            spectra[(lam, seed)] = eigs
            print(f"lam={lam} seed={seed}  top-5 eigs: {eigs[:5].round(2)}")

    return spectra


def plot_eigenvalue_spectra(spectra, save_path="eigenvalue_spectra.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for lam in LAMBDAS:
        seed_spectra = [spectra[(lam, s)] for s in SEEDS]
        mean_spec = np.mean(seed_spectra, axis=0)
        axes[0].semilogy(mean_spec, label=f"λ={lam}")

    axes[0].set_xlabel("Eigenvalue rank")
    axes[0].set_ylabel("Eigenvalue (log scale)")
    axes[0].set_title("Full eigenvalue spectrum")
    axes[0].legend()
    for lam in LAMBDAS:
        seed_spectra = [spectra[(lam, s)] for s in SEEDS]
        mean_spec = np.mean(seed_spectra, axis=0)
        cumvar = np.cumsum(mean_spec) / mean_spec.sum()
        axes[1].plot(cumvar, label=f"λ={lam}")

    axes[1].set_xlabel("Number of components")
    axes[1].set_ylabel("Cumulative variance explained")
    axes[1].set_title("Cumulative variance explained")
    axes[1].axhline(0.95, color="gray", linestyle="--", alpha=0.5, label="95%")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

def compute_time_resolved_pr(window_size=20, step_size=5):
    pr_curves = {}
    time_centers = None

    for lam in LAMBDAS:
        for seed in SEEDS:
            H = np.load(hidden_states_path(lam, seed))
            n_trials, T, n_units = H.shape

            centers = []
            prs = []

            for win_start in range(0, T - window_size + 1, step_size):
                win_end = win_start + window_size
                X = H[:, win_start:win_end, :].reshape(-1, n_units)
                prs.append(participation_ratio(X))
                centers.append((win_start + win_end) / 2 * DT) # center in ms

            pr_curves[(lam, seed)] = np.array(prs)
            if time_centers is None:
                time_centers = np.array(centers)

            print(f"lam={lam} seed={seed}  delay-mean PR={np.mean(prs[8:33]):.3f}")

    return time_centers, pr_curves


def plot_time_resolved_pr(time_centers, pr_curves, save_path="time_resolved_pr.png"):
    fig, ax = plt.subplots(figsize=(12, 5))

    for lam in LAMBDAS:
        curves = [pr_curves[(lam, s)] for s in SEEDS]
        mean_curve = np.mean(curves, axis=0)
        sem_curve = np.std(curves, axis=0, ddof=1) / np.sqrt(len(SEEDS))
        ax.plot(time_centers, mean_curve, label=f"λ={lam}")
        ax.fill_between(time_centers, mean_curve - sem_curve, mean_curve + sem_curve, alpha=0.15)

    # Shade periods
    period_spans = {
        "fixation": (0, 300), "sample": (300, 800),
        "delay": (800, 3300), "test": (3300, 3800), "decision": (3800, 4300)
    }
    colors = {"sample": "yellow", "delay": "lightgray", "test": "lightblue", "decision": "lightgreen"}
    for period, (start_ms, end_ms) in period_spans.items():
        if period in colors:
            ax.axvspan(start_ms, end_ms, alpha=0.2, color=colors[period])

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Participation Ratio")
    ax.set_title("Time-resolved PR across trial")
    ax.legend(loc="upper left")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

def compute_pc_correlations(lam, seed, n_pcs=10):

    H = np.load(hidden_states_path(lam, seed))
    infos = get_trial_metadata(lam, seed, n_trials=H.shape[0])
    n_trials = H.shape[0]

    delay = H[:, DELAY_START:DELAY_END, :]
    n_delay = DELAY_END - DELAY_START

    X = delay.reshape(-1, HIDDEN_DIM)
    X = X - X.mean(axis=0)

    cov = np.cov(X, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    explained_variance = eigenvalues[:n_pcs] / eigenvalues.sum()

    projections = X @ eigenvectors[:, :n_pcs]

    target_angles = np.array([info['target_angle'] for info in infos])
    target_sin = np.repeat(target_angles, n_delay)
    target_sin = np.sin(target_sin)
    target_cos = np.repeat(np.cos(target_angles), n_delay)

    time_in_delay = np.tile(np.linspace(0, 1, n_delay), n_trials)

    distractor_active = np.zeros(n_trials * n_delay)
    for trial_idx, info in enumerate(infos):
        for onset in info['distractor_onsets']:
            rel_start = onset - DELAY_START
            rel_end = min(rel_start + 10, n_delay) # 10 steps per slot
            if rel_start >= 0 and rel_start < n_delay:
                flat_start = trial_idx * n_delay + rel_start
                flat_end = trial_idx * n_delay + rel_end
                distractor_active[flat_start:flat_end] = 1.0

    correlations = {}
    variables = {
        'target_sin': target_sin,
        'target_cos': target_cos,
        'distractor_active': distractor_active,
        'time_in_delay': time_in_delay,
    }

    for var_name, var_values in variables.items():
        corrs = []
        for pc in range(n_pcs):
            r = np.corrcoef(projections[:, pc], var_values)[0, 1]
            corrs.append(r)
        correlations[var_name] = np.array(corrs)

    return correlations, explained_variance


def plot_pc_correlations(lam, seed, save_path=None):
    correlations, explained_var = compute_pc_correlations(lam, seed)

    n_pcs = len(explained_var)
    var_names = list(correlations.keys())
    corr_matrix = np.array([correlations[v] for v in var_names]) # (n_vars, n_pcs)

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n_pcs))
    ax.set_xticklabels([f"PC{i+1}\n({explained_var[i]:.1%})" for i in range(n_pcs)], fontsize=8)
    ax.set_yticks(range(len(var_names)))
    ax.set_yticklabels(var_names)
    ax.set_xlabel("Principal Component (% variance explained)")
    ax.set_title(f"PC-variable correlations — λ={lam}, seed={seed}")
    plt.colorbar(im, ax=ax, label="Pearson r")

    if save_path is None:
        save_path = f"pc_correlations_lam{lam}_seed{seed}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_pc_correlations_grid(save_path="pc_correlations_grid.png"):
    fig, axes = plt.subplots(len(LAMBDAS), 1, figsize=(10, 3 * len(LAMBDAS)))

    for ax_idx, lam in enumerate(LAMBDAS):
        correlations, explained_var = compute_pc_correlations(lam, seed=0)
        n_pcs = len(explained_var)
        var_names = list(correlations.keys())
        corr_matrix = np.array([correlations[v] for v in var_names])

        im = axes[ax_idx].imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        axes[ax_idx].set_xticks(range(n_pcs))
        axes[ax_idx].set_xticklabels([f"PC{i+1}" for i in range(n_pcs)], fontsize=7)
        axes[ax_idx].set_yticks(range(len(var_names)))
        axes[ax_idx].set_yticklabels(var_names, fontsize=8)
        axes[ax_idx].set_title(f"λ={lam}", fontsize=10)

    plt.tight_layout()
    plt.colorbar(im, ax=axes, label="Pearson r", shrink=0.6)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

def compute_memory_specific_pr():

    results = []

    for lam in LAMBDAS:
        for seed in SEEDS:
            H = np.load(hidden_states_path(lam, seed))
            infos = get_trial_metadata(lam, seed, n_trials=H.shape[0])
            n_trials = H.shape[0]
            n_delay = DELAY_END - DELAY_START

            delay = H[:, DELAY_START:DELAY_END, :] # (n_trials, 125, 128)
            X = delay.reshape(-1, HIDDEN_DIM)
            raw_pr = participation_ratio(X)

            if lam == 0:
                results.append((lam, seed, raw_pr, raw_pr))
                print(f"lam={lam} seed={seed}  raw_PR={raw_pr:.3f}  mem_PR={raw_pr:.3f}")
                continue

            distractor_active = np.zeros(n_trials * n_delay)
            recent_d_sin = np.zeros(n_trials * n_delay)
            recent_d_cos = np.zeros(n_trials * n_delay)

            for trial_idx, info in enumerate(infos):
                last_angle = None
                for t_rel in range(n_delay):
                    t_abs = DELAY_START + t_rel
                    flat_idx = trial_idx * n_delay + t_rel

                    for d_idx, onset in enumerate(info['distractor_onsets']):
                        if onset <= t_abs < onset + 10:
                            distractor_active[flat_idx] = 1.0
                            last_angle = info['distractor_angles'][d_idx]
                            break
                        elif t_abs >= onset + 10 and (last_angle is None or onset > DELAY_START):
                            if d_idx < len(info['distractor_angles']):
                                last_angle = info['distractor_angles'][d_idx]

                    if last_angle is not None:
                        recent_d_sin[flat_idx] = np.sin(last_angle)
                        recent_d_cos[flat_idx] = np.cos(last_angle)

            regressors = np.column_stack([
                distractor_active,
                recent_d_sin,
                recent_d_cos,
                np.ones(n_trials * n_delay),
            ])

            X_residual = X.copy()
            for unit in range(HIDDEN_DIM):
                beta, _, _, _ = np.linalg.lstsq(regressors, X[:, unit], rcond=None)
                X_residual[:, unit] = X[:, unit] - regressors @ beta

            memory_pr = participation_ratio(X_residual)
            results.append((lam, seed, raw_pr, memory_pr))
            print(f"lam={lam} seed={seed}  raw_PR={raw_pr:.3f}  mem_PR={memory_pr:.3f}")

    return results


def plot_memory_specific_pr(results, save_path="memory_specific_pr.png"):
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, idx, color in [("Raw PR", 2, "C0"), ("Memory-specific PR", 3, "C3")]:
        means, sems = [], []
        for lam in LAMBDAS:
            vals = [r[idx] for r in results if r[0] == lam]
            means.append(np.mean(vals))
            sems.append(np.std(vals, ddof=1) / np.sqrt(len(vals)))
        ax.errorbar(LAMBDAS, means, yerr=sems, marker="o", capsize=4, label=label, color=color)

    ax.set_xlabel("λ (training distractor intensity)")
    ax.set_ylabel("Participation Ratio")
    ax.set_title("Raw vs memory-specific PR")
    ax.legend()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def compute_on_manifold_variance(model_loader_fn, n_clean_components=10):

    X_clean, _, _ = generate_trials(num_trials=50, lam=0.0, seed=9999)
    results = []

    for lam in LAMBDAS:
        for seed in SEEDS:
            model = model_loader_fn(lam, seed)

            with torch.no_grad():
                x_tensor = torch.tensor(X_clean, dtype=torch.float32)
                _, H_clean = model(x_tensor)
                H_clean = H_clean.cpu().numpy()

            delay_clean = H_clean[:, DELAY_START:DELAY_END, :].reshape(-1, HIDDEN_DIM)
            delay_clean = delay_clean - delay_clean.mean(axis=0)
            cov_clean = np.cov(delay_clean, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_clean)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvectors = eigenvectors[:, idx]
            V_clean = eigenvectors[:, :n_clean_components]

            H_dist = np.load(hidden_states_path(lam, seed))
            delay_dist = H_dist[:, DELAY_START:DELAY_END, :].reshape(-1, HIDDEN_DIM)
            delay_dist_centered = delay_dist - delay_dist.mean(axis=0)

            total_var = np.sum(delay_dist_centered ** 2)
            projected = delay_dist_centered @ V_clean  
            reconstructed = projected @ V_clean.T 
            on_manifold_var = np.sum(reconstructed ** 2)

            fraction = on_manifold_var / total_var
            results.append((lam, seed, fraction))
            print(f"lam={lam} seed={seed}  on-manifold fraction={fraction:.3f}")

    return results


def plot_on_manifold_variance(results, save_path="on_manifold_variance.png"):
    fig, ax = plt.subplots(figsize=(8, 5))

    means, sems = [], []
    for lam in LAMBDAS:
        vals = [r[2] for r in results if r[0] == lam]
        means.append(np.mean(vals))
        sems.append(np.std(vals, ddof=1) / np.sqrt(len(vals)))

    ax.errorbar(LAMBDAS, means, yerr=sems, marker="o", capsize=4, color="C0")
    ax.set_xlabel("λ (training distractor intensity)")
    ax.set_ylabel("Fraction of variance in clean subspace")
    ax.set_title("On-manifold variance (clean PCA subspace)")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

from rnn_model import VanillaRNN

def load_model(lam, seed):
    model = VanillaRNN(input_size=33, hidden_size=128, output_size=3)
    
    if lam == 0:
        path = f'experiments/checkpoints/lam0_seed{seed}/epoch_100.pt'
    else:
        path = f'experiments/checkpoints/finetuned_lam{lam}_seed{seed}/target95.pt'
    
    ck = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    return model

if __name__ == "__main__":
    # print("=" * 60)
    # print("METHOD 2: Eigenvalue spectra")
    # print("=" * 60)
    # spectra = compute_eigenvalue_spectra()
    # plot_eigenvalue_spectra(spectra)

    # print("\n" + "=" * 60)
    # print("METHOD 3: Time-resolved PR")
    # print("=" * 60)
    # time_centers, pr_curves = compute_time_resolved_pr()
    # plot_time_resolved_pr(time_centers, pr_curves)

    # print("\n" + "=" * 60)
    # print("METHOD 4: PC-variable correlations")
    # print("=" * 60)
    # plot_pc_correlations_grid()

    # print("\n" + "=" * 60)
    # print("METHOD 5: Memory-specific PR")
    # print("=" * 60)
    # mem_results = compute_memory_specific_pr()
    # plot_memory_specific_pr(mem_results)

    print("\n" + "=" * 60)
    print("METHOD 1: PR on clean trials")
    print("=" * 60)
    clean_results = pr_on_clean_trials(load_model)
    original_results = [(lam, seed, participation_ratio(
        np.load(hidden_states_path(lam, seed))[:, DELAY_START:DELAY_END, :].reshape(-1, HIDDEN_DIM)
    )) for lam in LAMBDAS for seed in SEEDS]
    plot_pr_clean_vs_original(clean_results, original_results)

    # print("\n" + "=" * 60)
    # print("METHOD 6: On-manifold analysis")
    # print("=" * 60)
    # on_manifold_results = compute_on_manifold_variance(load_model)
    # plot_on_manifold_variance(on_manifold_results)
