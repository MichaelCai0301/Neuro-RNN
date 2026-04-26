import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

from tasks.poisson_dms import generate_trials


CHECKPOINT_ROOT = Path("experiments/checkpoints")
DELAY_START = 40
DELAY_END = 165
SAMPLE_START = 15
SAMPLE_END = 40
DT = 20
LAMBDAS = [0, 2, 4, 6, 8, 10]
SEEDS = [0, 1, 2]
HIDDEN_DIM = 128
N_PCS = 10
CORRELATION_THRESHOLD = 0.3
DISTRACTOR_SLOT_STEPS = 10 # 200ms / 20ms per step

OUT_DIR = Path("analysis_results/pc_analysis")


def ensure_out_dir():
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def hidden_states_path(lam, seed):
    if lam == 0:
        return CHECKPOINT_ROOT / f"lam0_seed{seed}" / "hidden_states_val.npy"
    return CHECKPOINT_ROOT / f"finetuned_lam{lam}_seed{seed}" / "hidden_states_val_matched.npy"


def get_trial_metadata(lam, seed, n_trials=50):
    val_seed = seed * 1000 + 99
    _, _, infos = generate_trials(num_trials=500, lam=lam, seed=val_seed)
    return infos[:n_trials]


def compute_pc_correlations(lam, seed, n_pcs=N_PCS):
    H = np.load(hidden_states_path(lam, seed))
    infos = get_trial_metadata(lam, seed, n_trials=H.shape[0])
    n_trials = H.shape[0]
    n_delay = DELAY_END - DELAY_START
    delay = H[:, DELAY_START:DELAY_END, :]
    X = delay.reshape(-1, HIDDEN_DIM)
    X_centered = X - X.mean(axis=0)
    cov = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    explained_variance = eigenvalues[:n_pcs] / eigenvalues.sum()

    projections = X_centered @ eigenvectors[:, :n_pcs]

    target_angles = np.array([info['target_angle'] for info in infos])
    target_sin = np.repeat(np.sin(target_angles), n_delay)
    target_cos = np.repeat(np.cos(target_angles), n_delay)
    time_in_delay = np.tile(np.linspace(0, 1, n_delay), n_trials)

    distractor_active = np.zeros(n_trials * n_delay)
    for trial_idx, info in enumerate(infos):
        for onset in info['distractor_onsets']:
            rel_start = onset - DELAY_START
            rel_end = min(rel_start + DISTRACTOR_SLOT_STEPS, n_delay)
            if 0 <= rel_start < n_delay:
                flat_start = trial_idx * n_delay + rel_start
                flat_end = trial_idx * n_delay + rel_end
                distractor_active[flat_start:flat_end] = 1.0

    recent_d_sin = np.zeros(n_trials * n_delay)
    recent_d_cos = np.zeros(n_trials * n_delay)
    for trial_idx, info in enumerate(infos):
        last_angle = None
        sorted_indices = np.argsort(info['distractor_onsets']) if len(info['distractor_onsets']) > 0 else []
        sorted_onsets = info['distractor_onsets'][sorted_indices] if len(sorted_indices) > 0 else []
        sorted_angles = info['distractor_angles'][sorted_indices] if len(sorted_indices) > 0 else []

        d_ptr = 0
        for t_rel in range(n_delay):
            t_abs = DELAY_START + t_rel
            while d_ptr < len(sorted_onsets) and sorted_onsets[d_ptr] + DISTRACTOR_SLOT_STEPS <= t_abs:
                last_angle = sorted_angles[d_ptr]
                d_ptr += 1
            if d_ptr < len(sorted_onsets) and sorted_onsets[d_ptr] <= t_abs < sorted_onsets[d_ptr] + DISTRACTOR_SLOT_STEPS:
                last_angle = sorted_angles[d_ptr]

            flat_idx = trial_idx * n_delay + t_rel
            if last_angle is not None:
                recent_d_sin[flat_idx] = np.sin(last_angle)
                recent_d_cos[flat_idx] = np.cos(last_angle)

    variables = {
        'target_sin': target_sin,
        'target_cos': target_cos,
        'distractor_active': distractor_active,
        'recent_distractor_sin': recent_d_sin,
        'recent_distractor_cos': recent_d_cos,
        'time_in_delay': time_in_delay,
    }

    correlations = {}
    for var_name, var_values in variables.items():
        corrs = []
        for pc in range(n_pcs):
            if np.std(var_values) < 1e-10:
                corrs.append(0.0)
            else:
                r = np.corrcoef(projections[:, pc], var_values)[0, 1]
                corrs.append(r if np.isfinite(r) else 0.0)
        correlations[var_name] = np.array(corrs)

    return correlations, explained_variance, eigenvalues, eigenvectors

def plot_pc_correlations_grid(seed=0, save_path=None):
    ensure_out_dir()
    if save_path is None:
        save_path = OUT_DIR / f"pc_correlations_grid_seed{seed}.png"

    all_correlations = {}
    all_explained_var = {}

    fig, axes = plt.subplots(len(LAMBDAS), 1, figsize=(12, 3 * len(LAMBDAS)))

    for ax_idx, lam in enumerate(LAMBDAS):
        correlations, explained_var, _, _ = compute_pc_correlations(lam, seed)
        all_correlations[(lam, seed)] = correlations
        all_explained_var[(lam, seed)] = explained_var

        var_names = list(correlations.keys())
        corr_matrix = np.array([correlations[v] for v in var_names])

        im = axes[ax_idx].imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        axes[ax_idx].set_xticks(range(N_PCS))
        axes[ax_idx].set_xticklabels(
            [f"PC{i+1}\n({explained_var[i]:.1%})" for i in range(N_PCS)], fontsize=7
        )
        axes[ax_idx].set_yticks(range(len(var_names)))
        axes[ax_idx].set_yticklabels([v.replace('_', '\n') for v in var_names], fontsize=8)
        axes[ax_idx].set_title(f"λ = {lam}", fontsize=11, fontweight='bold')

    plt.colorbar(im, ax=axes, label="Pearson r", shrink=0.6, pad=0.02)
    plt.suptitle(f"PC-variable correlations across λ (seed {seed})", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

    return all_correlations, all_explained_var


def count_significant_pcs(correlations, threshold=CORRELATION_THRESHOLD):
    counts = {}
    for var_name, corrs in correlations.items():
        counts[var_name] = int(np.sum(np.abs(corrs) > threshold))
    return counts


def analysis_a_dimension_counts(save_path=None):
    ensure_out_dir()
    if save_path is None:
        save_path = OUT_DIR / "analysis_a_dimension_counts.png"

    target_counts = {lam: [] for lam in LAMBDAS}
    distractor_counts = {lam: [] for lam in LAMBDAS}
    total_significant = {lam: [] for lam in LAMBDAS}

    for lam in LAMBDAS:
        for seed in SEEDS:
            correlations, _, _, _ = compute_pc_correlations(lam, seed)
            counts = count_significant_pcs(correlations)

            target_pcs = set()
            for var in ['target_sin', 'target_cos']:
                for i, r in enumerate(correlations[var]):
                    if abs(r) > CORRELATION_THRESHOLD:
                        target_pcs.add(i)
            target_counts[lam].append(len(target_pcs))

            distractor_pcs = set()
            for var in ['distractor_active', 'recent_distractor_sin', 'recent_distractor_cos']:
                for i, r in enumerate(correlations[var]):
                    if abs(r) > CORRELATION_THRESHOLD:
                        distractor_pcs.add(i)
            distractor_counts[lam].append(len(distractor_pcs))

            all_significant = target_pcs | distractor_pcs
            for i, r in enumerate(correlations['time_in_delay']):
                if abs(r) > CORRELATION_THRESHOLD:
                    all_significant.add(i)
            total_significant[lam].append(len(all_significant))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for label, data, color, marker in [
        ("Target-correlated PCs", target_counts, "C0", "o"),
        ("Distractor-correlated PCs", distractor_counts, "C3", "s"),
        ("Total significant PCs", total_significant, "C2", "^"),
    ]:
        means = [np.mean(data[lam]) for lam in LAMBDAS]
        sems = [np.std(data[lam], ddof=1) / np.sqrt(len(SEEDS)) if len(data[lam]) > 1 else 0
                for lam in LAMBDAS]
        axes[0].errorbar(LAMBDAS, means, yerr=sems, marker=marker, capsize=4,
                         label=label, color=color)

    axes[0].set_xlabel("λ (training distractor intensity)")
    axes[0].set_ylabel(f"Number of PCs (|r| > {CORRELATION_THRESHOLD})")
    axes[0].set_title("Significant PCs per function")
    axes[0].legend()
    axes[0].set_ylim(bottom=0)

    ratios = {lam: [] for lam in LAMBDAS}
    for lam in LAMBDAS:
        for i in range(len(SEEDS)):
            t = target_counts[lam][i]
            d = distractor_counts[lam][i]
            ratios[lam].append(d / max(t, 1))

    means = [np.mean(ratios[lam]) for lam in LAMBDAS]
    sems = [np.std(ratios[lam], ddof=1) / np.sqrt(len(SEEDS)) if len(ratios[lam]) > 1 else 0
            for lam in LAMBDAS]
    axes[1].errorbar(LAMBDAS, means, yerr=sems, marker="o", capsize=4, color="C1")
    axes[1].set_xlabel("λ")
    axes[1].set_ylabel("Distractor PCs / Target PCs")
    axes[1].set_title("Ratio of distractor to target dimensions")
    axes[1].axhline(1.0, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

    return target_counts, distractor_counts

def compute_subspace_overlap(correlations, threshold=CORRELATION_THRESHOLD):
    target_pcs = set()
    for var in ['target_sin', 'target_cos']:
        for i, r in enumerate(correlations[var]):
            if abs(r) > threshold:
                target_pcs.add(i)

    distractor_pcs = set()
    for var in ['distractor_active', 'recent_distractor_sin', 'recent_distractor_cos']:
        for i, r in enumerate(correlations[var]):
            if abs(r) > threshold:
                distractor_pcs.add(i)

    shared = target_pcs & distractor_pcs
    return len(target_pcs), len(distractor_pcs), len(shared)


def analysis_b_subspace_overlap(save_path=None):

    ensure_out_dir()
    if save_path is None:
        save_path = OUT_DIR / "analysis_b_subspace_overlap.png"

    shared_counts = {lam: [] for lam in LAMBDAS}
    target_only_counts = {lam: [] for lam in LAMBDAS}
    distractor_only_counts = {lam: [] for lam in LAMBDAS}

    for lam in LAMBDAS:
        for seed in SEEDS:
            correlations, _, _, _ = compute_pc_correlations(lam, seed)
            n_target, n_distractor, n_shared = compute_subspace_overlap(correlations)
            shared_counts[lam].append(n_shared)
            target_only_counts[lam].append(n_target - n_shared)
            distractor_only_counts[lam].append(n_distractor - n_shared)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(LAMBDAS))
    width = 0.6

    target_means = [np.mean(target_only_counts[lam]) for lam in LAMBDAS]
    distractor_means = [np.mean(distractor_only_counts[lam]) for lam in LAMBDAS]
    shared_means = [np.mean(shared_counts[lam]) for lam in LAMBDAS]

    ax.bar(x, target_means, width, label="Target-only PCs", color="C0", alpha=0.8)
    ax.bar(x, shared_means, width, bottom=target_means, label="Shared PCs", color="C4", alpha=0.8)
    ax.bar(x, distractor_means, width,
           bottom=[t + s for t, s in zip(target_means, shared_means)],
           label="Distractor-only PCs", color="C3", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"λ={lam}" for lam in LAMBDAS])
    ax.set_ylabel(f"Number of PCs (|r| > {CORRELATION_THRESHOLD})")
    ax.set_title("Functional overlap between target and distractor subspaces")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

    return shared_counts

def compute_variance_partition(correlations, explained_variance, threshold=CORRELATION_THRESHOLD):
 
    target_var = 0.0
    distractor_var = 0.0
    shared_var = 0.0
    time_var = 0.0
    other_var = 0.0

    for i, ev in enumerate(explained_variance):
        is_target = any(abs(correlations[v][i]) > threshold for v in ['target_sin', 'target_cos'])
        is_distractor = any(abs(correlations[v][i]) > threshold
                           for v in ['distractor_active', 'recent_distractor_sin', 'recent_distractor_cos'])
        is_time = abs(correlations['time_in_delay'][i]) > threshold

        if is_target and is_distractor:
            shared_var += ev
        elif is_target:
            target_var += ev
        elif is_distractor:
            distractor_var += ev
        elif is_time:
            time_var += ev
        else:
            other_var += ev

    return {
        'target': target_var,
        'distractor': distractor_var,
        'shared': shared_var,
        'time': time_var,
        'other': other_var,
    }


def analysis_c_variance_partitioning(save_path=None):

    ensure_out_dir()
    if save_path is None:
        save_path = OUT_DIR / "analysis_c_variance_partitioning.png"

    partitions = {lam: {'target': [], 'distractor': [], 'shared': [], 'time': [], 'other': []}
                  for lam in LAMBDAS}

    for lam in LAMBDAS:
        for seed in SEEDS:
            correlations, explained_var, _, _ = compute_pc_correlations(lam, seed)
            partition = compute_variance_partition(correlations, explained_var)
            for key in partition:
                partitions[lam][key].append(partition[key])

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(LAMBDAS))
    width = 0.6

    categories = ['target', 'distractor', 'shared', 'time', 'other']
    colors = {'target': 'C0', 'distractor': 'C3', 'shared': 'C4', 'time': 'C2', 'other': 'C7'}
    labels = {'target': 'Target memory', 'distractor': 'Distractor processing',
              'shared': 'Shared (target+distractor)', 'time': 'Temporal context', 'other': 'Unattributed'}

    bottoms = np.zeros(len(LAMBDAS))
    for cat in categories:
        means = [np.mean(partitions[lam][cat]) for lam in LAMBDAS]
        ax.bar(x, means, width, bottom=bottoms, label=labels[cat], color=colors[cat], alpha=0.85)
        bottoms += means

    ax.set_xticks(x)
    ax.set_xticklabels([f"λ={lam}" for lam in LAMBDAS])
    ax.set_ylabel("Fraction of total variance explained")
    ax.set_title("Variance partitioning by functional role")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, min(1.0, bottoms.max() * 1.1))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

    return partitions


def compute_target_info_rank(correlations):
    combined = np.sqrt(correlations['target_sin'] ** 2 + correlations['target_cos'] ** 2)
    return np.argsort(combined)[::-1]


def analysis_d_rank_shift(save_path=None):
    ensure_out_dir()
    if save_path is None:
        save_path = OUT_DIR / "analysis_d_rank_shift.png"

    top1_ranks = {lam: [] for lam in LAMBDAS}
    top2_ranks = {lam: [] for lam in LAMBDAS}
    target_strengths = {lam: [] for lam in LAMBDAS}

    for lam in LAMBDAS:
        for seed in SEEDS:
            correlations, explained_var, _, _ = compute_pc_correlations(lam, seed)
            ranked = compute_target_info_rank(correlations)
            top1_ranks[lam].append(ranked[0] + 1) # 1-indexed for readability
            top2_ranks[lam].append(ranked[1] + 1)

            combined = np.sqrt(correlations['target_sin'] ** 2 + correlations['target_cos'] ** 2)
            target_strengths[lam].append(combined.sum())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for label, data, color, marker in [
        ("Strongest target PC", top1_ranks, "C0", "o"),
        ("2nd strongest target PC", top2_ranks, "C1", "s"),
    ]:
        means = [np.mean(data[lam]) for lam in LAMBDAS]
        sems = [np.std(data[lam], ddof=1) / np.sqrt(len(SEEDS)) if len(data[lam]) > 1 else 0
                for lam in LAMBDAS]
        axes[0].errorbar(LAMBDAS, means, yerr=sems, marker=marker, capsize=4,
                         label=label, color=color)

    axes[0].set_xlabel("λ")
    axes[0].set_ylabel("PC rank (1 = highest variance)")
    axes[0].set_title("Which PCs carry target information?")
    axes[0].legend()
    axes[0].invert_yaxis()

    means = [np.mean(target_strengths[lam]) for lam in LAMBDAS]
    sems = [np.std(target_strengths[lam], ddof=1) / np.sqrt(len(SEEDS))
            if len(target_strengths[lam]) > 1 else 0 for lam in LAMBDAS]
    axes[1].errorbar(LAMBDAS, means, yerr=sems, marker="o", capsize=4, color="C0")
    axes[1].set_xlabel("λ")
    axes[1].set_ylabel("Σ √(r²_sin + r²_cos) across PCs")
    axes[1].set_title("Total target information strength")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

    return top1_ranks, top2_ranks


def compute_target_encoding_vector(correlations):
    return np.concatenate([correlations['target_sin'], correlations['target_cos']])


def analysis_e_encoding_similarity(seed=0, save_path=None):

    ensure_out_dir()
    if save_path is None:
        save_path = OUT_DIR / f"analysis_e_encoding_similarity_seed{seed}.png"

    encoding_vectors = {}
    for lam in LAMBDAS:
        correlations, _, _, _ = compute_pc_correlations(lam, seed)
        encoding_vectors[lam] = compute_target_encoding_vector(correlations)

    n = len(LAMBDAS)
    sim_matrix = np.zeros((n, n))
    for i, lam_i in enumerate(LAMBDAS):
        for j, lam_j in enumerate(LAMBDAS):
            vi = encoding_vectors[lam_i]
            vj = encoding_vectors[lam_j]
            norm_i = np.linalg.norm(vi)
            norm_j = np.linalg.norm(vj)
            if norm_i > 1e-10 and norm_j > 1e-10:
                sim_matrix[i, j] = np.dot(vi, vj) / (norm_i * norm_j)
            else:
                sim_matrix[i, j] = 0.0

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(sim_matrix, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels([f"λ={lam}" for lam in LAMBDAS])
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"λ={lam}" for lam in LAMBDAS])
    ax.set_title(f"Target encoding similarity across λ (seed {seed})")

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{sim_matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=9, color="black" if abs(sim_matrix[i, j]) < 0.7 else "white")

    plt.colorbar(im, ax=ax, label="Cosine similarity", shrink=0.8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

    return sim_matrix

def compute_subspace_orthogonality(correlations, eigenvectors, threshold=CORRELATION_THRESHOLD):
    target_indices = set()
    for var in ['target_sin', 'target_cos']:
        for i, r in enumerate(correlations[var]):
            if abs(r) > threshold:
                target_indices.add(i)

    distractor_indices = set()
    for var in ['distractor_active', 'recent_distractor_sin', 'recent_distractor_cos']:
        for i, r in enumerate(correlations[var]):
            if abs(r) > threshold:
                distractor_indices.add(i)

    target_only = sorted(target_indices - distractor_indices)
    distractor_only = sorted(distractor_indices - target_indices)

    if len(target_only) == 0 or len(distractor_only) == 0:
        return np.array([]), 90.0

    U_target = eigenvectors[:, target_only] 
    U_distractor = eigenvectors[:, distractor_only]

    M = U_target.T @ U_distractor 
    singular_values = np.linalg.svd(M, compute_uv=False)
    singular_values = np.clip(singular_values, 0, 1)
    principal_angles = np.degrees(np.arccos(singular_values))

    return principal_angles, np.mean(principal_angles)


def analysis_subspace_orthogonality(save_path=None):

    ensure_out_dir()
    if save_path is None:
        save_path = OUT_DIR / "subspace_orthogonality.png"

    mean_angles = {lam: [] for lam in LAMBDAS}

    for lam in LAMBDAS:
        for seed in SEEDS:
            correlations, _, eigenvalues, eigenvectors = compute_pc_correlations(lam, seed)
            _, mean_angle = compute_subspace_orthogonality(correlations, eigenvectors)
            mean_angles[lam].append(mean_angle)
            print(f"lam={lam} seed={seed}  mean principal angle={mean_angle:.1f}°")

    fig, ax = plt.subplots(figsize=(8, 5))
    means = [np.mean(mean_angles[lam]) for lam in LAMBDAS]
    sems = [np.std(mean_angles[lam], ddof=1) / np.sqrt(len(SEEDS))
            if len(mean_angles[lam]) > 1 else 0 for lam in LAMBDAS]
    ax.errorbar(LAMBDAS, means, yerr=sems, marker="o", capsize=4, color="C0")
    ax.set_xlabel("λ (training distractor intensity)")
    ax.set_ylabel("Mean principal angle (degrees)")
    ax.set_title("Orthogonality between target and distractor subspaces")
    ax.axhline(90, color="gray", linestyle="--", alpha=0.5, label="Perfect orthogonality")
    ax.set_ylim(0, 95)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

    return mean_angles

import csv

def generate_summary_table(save_path=None):

    ensure_out_dir()
    if save_path is None:
        save_path = OUT_DIR / "summary_table.csv"

    rows = []
    header = [
        "lam", "seed",
        "n_target_pcs", "n_distractor_pcs", "n_shared_pcs",
        "var_target", "var_distractor", "var_shared", "var_other",
        "top_target_pc_rank", "mean_principal_angle"
    ]

    for lam in LAMBDAS:
        for seed in SEEDS:
            correlations, explained_var, eigenvalues, eigenvectors = compute_pc_correlations(lam, seed)

            n_target, n_distractor, n_shared = compute_subspace_overlap(correlations)
            partition = compute_variance_partition(correlations, explained_var)
            ranked = compute_target_info_rank(correlations)
            _, mean_angle = compute_subspace_orthogonality(correlations, eigenvectors)

            rows.append([
                lam, seed,
                n_target, n_distractor, n_shared,
                f"{partition['target']:.4f}",
                f"{partition['distractor']:.4f}",
                f"{partition['shared']:.4f}",
                f"{partition['other']:.4f}",
                ranked[0] + 1,
                f"{mean_angle:.1f}",
            ])

    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)

    print(f"Saved: {save_path}")

    print(f"\n{'lam':>4} {'seed':>4} {'tgt':>4} {'dst':>4} {'shrd':>4} "
          f"{'v_tgt':>7} {'v_dst':>7} {'v_shrd':>7} {'v_oth':>7} "
          f"{'top_pc':>6} {'angle':>6}")
    print("-" * 75)
    for row in rows:
        print(f"{row[0]:>4} {row[1]:>4} {row[2]:>4} {row[3]:>4} {row[4]:>4} "
              f"{row[5]:>7} {row[6]:>7} {row[7]:>7} {row[8]:>7} "
              f"{row[9]:>6} {row[10]:>6}")


def run_all():
    # Included A/B/C, D/E need to be more polished
    ensure_out_dir()

    print("=" * 70)
    print("PC CORRELATION GRID")
    print("=" * 70)
    for seed in SEEDS:
        plot_pc_correlations_grid(seed=seed)

    print("\n" + "=" * 70)
    print("ANALYSIS A: Dimension counts per function")
    print("=" * 70)
    analysis_a_dimension_counts()

    print("\n" + "=" * 70)
    print("ANALYSIS B: Subspace overlap")
    print("=" * 70)
    analysis_b_subspace_overlap()

    print("\n" + "=" * 70)
    print("ANALYSIS C: Variance partitioning")
    print("=" * 70)
    analysis_c_variance_partitioning()

    print("\n" + "=" * 70)
    print("ANALYSIS D: PC rank shift")
    print("=" * 70)
    analysis_d_rank_shift()

    print("\n" + "=" * 70)
    print("ANALYSIS E: Cross-lambda encoding similarity")
    print("=" * 70)
    for seed in SEEDS:
        analysis_e_encoding_similarity(seed=seed)

    print("\n" + "=" * 70)
    print("SUBSPACE ORTHOGONALITY")
    print("=" * 70)
    analysis_subspace_orthogonality()

    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    generate_summary_table()

    print("\n" + "=" * 70)
    print(f"ALL DONE. Results saved to {OUT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    run_all()
