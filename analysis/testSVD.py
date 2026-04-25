"""
SVD on the learning-induced perturbation Delta_W = W_trained - W_initial.

Following Schuessler et al. (2020), Eq. (1) and (4): the trained recurrent
weight matrix is decomposed as W = W_0 + Delta_W, and SVD is applied to
Delta_W to isolate the structure that emerges from learning rather than the
random component carried over from initialization.

Lineage on disk: each finetuned_lam{L}_seed{s}/target95.pt was forked from
lam0_seed{s}/epoch_100.pt, so W_0 for each finetuned network is exactly the
matched-seed baseline's W_hh.

lambda=0 has no comparable perturbation (the baseline IS the reference), so
this analysis covers lambda in {2, 4, 6, 8, 10} only.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path

CHECKPOINT_ROOT = Path(__file__).resolve().parent.parent / "experiments" / "checkpoints"
LAMBDAS = [2, 4, 6, 8, 10]
SEEDS = [0, 1, 2]


def load_W(path: Path) -> np.ndarray:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt["W_hh"]


def main():
    spectra_by_lam = {l: [] for l in LAMBDAS}

    for seed in SEEDS:
        W0 = load_W(CHECKPOINT_ROOT / f"lam0_seed{seed}" / "epoch_100.pt")
        for lam in LAMBDAS:
            W_trained = load_W(CHECKPOINT_ROOT / f"finetuned_lam{lam}_seed{seed}" / "target95.pt")
            dW = W_trained - W0
            s = np.linalg.svd(dW, compute_uv=False)
            spectra_by_lam[lam].append(s)
            frob = np.linalg.norm(dW)
            print(f"lam={lam} seed={seed}  ||ΔW||_F={frob:.3f}  top SV={s[0]:.3f}")

    out_dir = Path(__file__).resolve().parent

    fig, ax = plt.subplots(figsize=(7, 5))
    for lam in LAMBDAS:
        s_mean = np.mean(np.array(spectra_by_lam[lam]), axis=0)
        ax.plot(np.arange(1, s_mean.size + 1), s_mean, label=f"λ={lam}", marker=".", markersize=3)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("Singular-value index")
    ax.set_ylabel("Singular value of ΔW")
    ax.set_title("Singular-value spectrum of learning-induced perturbation ΔW")
    ax.legend()
    fig.savefig(out_dir / "svd_spectrum_vs_lambda.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_dir / 'svd_spectrum_vs_lambda.png'}")


if __name__ == "__main__":
    main()
