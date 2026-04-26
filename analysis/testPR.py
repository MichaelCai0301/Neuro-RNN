"""
Participation-ratio PCA on delay-period hidden states.

One PR scalar per network, then plot PR vs lambda across seeds.
See README.md for the data layout.
"""

import csv

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- config ---------------------------------------------------------------

CHECKPOINT_ROOT = Path(__file__).resolve().parent.parent / "experiments" / "checkpoints"

# Trial timing (from methods / README):
#   fixation [0, 15), sample [15, 40), delay [40, 165), test [165, 190), decision [190, 215)
DELAY_START = 40
DELAY_END = 165

LAMBDAS = [0, 2, 4, 6, 8, 10]
SEEDS = [0, 1, 2]


def hidden_states_path(lam: int, seed: int) -> Path:
    """Return the .npy file to load for a given (lambda, seed)."""
    if lam == 0:
        # lambda=0 is the baseline; no `_matched` variant exists — it IS the reference.
        return CHECKPOINT_ROOT / f"lam0_seed{seed}" / "hidden_states_val.npy"
    return CHECKPOINT_ROOT / f"finetuned_lam{lam}_seed{seed}" / "hidden_states_val_matched.npy"


# --- PCA / participation ratio -------------------------------------------

def participation_ratio(hidden_states: np.ndarray) -> float:
    """
    hidden_states: shape (n_trials, T, H) loaded straight from the .npy
    Returns: PR scalar in [1, H]

    Steps (maps to the 6 slide steps):
      1. Slice to the delay window and reshape to (n_trials * T_delay, H).
      2. Center each column (subtract per-unit mean). Do NOT z-score.
      3. Compute the H x H covariance matrix.
      4. Eigenvalues via np.linalg.eigh.
      5. PR = (sum eigs)^2 / sum(eigs^2).
    """
    # TODO: step 1 — slice [:, DELAY_START:DELAY_END, :] and reshape
    X = hidden_states[:, DELAY_START:DELAY_END, :].reshape(-1, hidden_states.shape[-1])
    # TODO: step 2 — center columns
    X = X - X.mean(axis=0)
    # TODO: step 3 — covariance (hint: np.cov with rowvar=False, or X.T @ X / (N-1))
    cov = np.cov(X, rowvar=False)
    # TODO: step 4 — eigenvalues
    eigs = np.linalg.eigh(cov)[0]
    # TODO: step 5 — return PR
    return np.sum(eigs)**2 / np.sum(eigs**2)


# --- driver ---------------------------------------------------------------

def main():
    results = []  # rows of (lam, seed, PR)

    for lam in LAMBDAS:
        for seed in SEEDS:
            path = hidden_states_path(lam, seed)
            h = np.load(path)
            print(h.shape)
            # TODO: print h.shape on the first iteration to confirm (n_trials, T, H)
            pr = participation_ratio(h)
            results.append((lam, seed, pr))
            print(f"lam={lam} seed={seed}  PR={pr:.3f}")
        

    # --- aggregate ---------------------------------------------------------
    # `results` is a list of (lam, seed, pr) tuples.
    # Goal: for each lambda, get mean and SEM across seeds.
    #
    # SEM = standard error of the mean = std(ddof=1) / sqrt(n_seeds)
    # With only 3 seeds it's a rough estimate, but it's what we have.

    means = []  # one entry per lambda, same order as LAMBDAS
    sems = []
    for lam in LAMBDAS:
        # TODO: pull out the PR values for this lambda (list-comp over `results`)
        prs_this_lam = [pr for l, s, pr in results if l == lam]                                               
        means.append(np.mean(prs_this_lam))
        sems.append(np.std(prs_this_lam, ddof=1) / np.sqrt(len(SEEDS)))

    # --- plot --------------------------------------------------------------
    out_dir = Path(__file__).resolve().parent
    plot_path = out_dir / "pr_vs_lambda.png"

    # TODO: plt.errorbar(LAMBDAS, means, yerr=sems, marker="o", capsize=4)
    plt.errorbar(LAMBDAS, means, yerr=sems, marker="o", capsize=4)
    plt.xlabel("lambda (distractor intensity)")
    plt.ylabel("Participation ratio")
    plt.title("Delay-period dimensionality vs distractor intensity")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved plot -> {plot_path}")
  

    # --- csv ---------------------------------------------------------------
    csv_path = out_dir / "pr_results.csv"
    # TODO: open csv_path for writing, write a header row ("lam", "seed", "pr"),
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lam", "seed", "pr"])
        for row in results:
            writer.writerow(row)
 
    print(f"saved csv  -> {csv_path}")


if __name__ == "__main__":
    main()
