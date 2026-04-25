"""
Eigenvalue-spectrum / variance-explained curves on delay-period hidden states.

Complementary to testPR.py: PR collapses the spectrum into one number, this
shows the shape. One curve per lambda, averaged across seeds.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

CHECKPOINT_ROOT = Path(__file__).resolve().parent.parent / "experiments" / "checkpoints"
DELAY_START = 40
DELAY_END = 165
LAMBDAS = [0, 2, 4, 6, 8, 10]
SEEDS = [0, 1, 2]


def hidden_states_path(lam: int, seed: int) -> Path:
    if lam == 0:
        return CHECKPOINT_ROOT / f"lam0_seed{seed}" / "hidden_states_val.npy"
    return CHECKPOINT_ROOT / f"finetuned_lam{lam}_seed{seed}" / "hidden_states_val_matched.npy"


def eigenvalue_spectrum(hidden_states: np.ndarray) -> np.ndarray:
    """Return the H eigenvalues of the delay-period covariance, sorted descending."""
    X = hidden_states[:, DELAY_START:DELAY_END, :].reshape(-1, hidden_states.shape[-1])
    X = X - X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    eigs = np.linalg.eigh(cov)[0]
    return eigs[::-1]


def main():
    spectra_by_lam = {}
    for lam in LAMBDAS:
        seeds_spectra = []
        for seed in SEEDS:
            h = np.load(hidden_states_path(lam, seed))
            seeds_spectra.append(eigenvalue_spectrum(h))
        spectra_by_lam[lam] = np.array(seeds_spectra)  # (n_seeds, H)

    out_dir = Path(__file__).resolve().parent

    # --- cumulative variance explained -------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    for lam in LAMBDAS:
        sp = spectra_by_lam[lam]
        cum = np.cumsum(sp, axis=1) / sp.sum(axis=1, keepdims=True)
        mean_cum = cum.mean(axis=0)
        ax.plot(np.arange(1, mean_cum.size + 1), mean_cum, label=f"λ={lam}", marker=".", markersize=3)
    ax.set_xscale("log")
    ax.set_xlabel("# of PCs")
    ax.set_ylabel("Cumulative variance explained")
    ax.set_title("Eigenvalue spectrum of delay-period activity")
    ax.legend()
    ax.set_ylim(0, 1.02)
    fig.savefig(out_dir / "spectrum_vs_lambda.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_dir / 'spectrum_vs_lambda.png'}")

    # --- raw eigenvalues (log scale) ---------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    for lam in LAMBDAS:
        sp = spectra_by_lam[lam].mean(axis=0)
        ax.plot(np.arange(1, sp.size + 1), sp, label=f"λ={lam}", marker=".", markersize=3)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("PC index")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("Raw eigenvalue spectrum (mean across seeds)")
    ax.legend()
    fig.savefig(out_dir / "eigenvalues_vs_lambda.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_dir / 'eigenvalues_vs_lambda.png'}")


if __name__ == "__main__":
    main()
