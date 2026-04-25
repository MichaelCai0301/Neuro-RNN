"""
Random unit-ablation analysis.

For each network: load the trained model, regenerate the same 50 validation
trials, then for each ablation fraction k, run the network forward with k*H
randomly chosen hidden units zeroed at every timestep. Repeat several times
with different random ablation sets, average the resulting decision accuracy.

Question: does training under stronger distraction make the network more or
less robust to losing units? If high-lambda networks rely on a few critical
units, ablating those should hurt them more; if they distribute memory
across many units, performance should degrade more gracefully.
"""

import csv
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rnn_model import VanillaRNN
from tasks.poisson_dms import generate_trials

CHECKPOINT_ROOT = REPO_ROOT / "experiments" / "checkpoints"
LAMBDAS = [0, 2, 4, 6, 8, 10]
SEEDS = [0, 1, 2]
ABLATION_FRACTIONS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
N_REPEATS = 5
HIDDEN_SIZE = 128


def checkpoint_path(lam: int, seed: int) -> Path:
    if lam == 0:
        return CHECKPOINT_ROOT / f"lam0_seed{seed}" / "epoch_100.pt"
    return CHECKPOINT_ROOT / f"finetuned_lam{lam}_seed{seed}" / "target95.pt"


def load_model(path: Path) -> VanillaRNN:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = VanillaRNN()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def forward_with_ablation(model: VanillaRNN, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Run the recurrent dynamics with `mask` (shape (H,)) applied to the hidden state every step."""
    batch_size, seq_len, _ = x.shape
    h = torch.zeros(batch_size, model.hidden_size)
    outs = []
    for t in range(seq_len):
        h = (1 - model.alpha) * h + model.alpha * torch.tanh(
            model.W_in(x[:, t]) + model.W_rec(h)
        )
        h = h * mask
        outs.append(model.readout(h))
    return torch.stack(outs, dim=1)


def decision_accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    pred_classes = predictions.argmax(dim=-1)
    is_dec = labels >= 1
    if not is_dec.any():
        return 0.0
    correct = ((pred_classes == labels) & is_dec).sum().float()
    return float(correct / is_dec.sum())


def main():
    rng = np.random.default_rng(42)
    results = []   # (lam, seed, fraction, accuracy)

    for lam in LAMBDAS:
        for seed in SEEDS:
            model = load_model(checkpoint_path(lam, seed))
            X, Y, _ = generate_trials(num_trials=500, lam=lam, seed=seed * 1000 + 99)
            X = torch.tensor(X[:50], dtype=torch.float32)
            Y = torch.tensor(Y[:50], dtype=torch.long)

            for frac in ABLATION_FRACTIONS:
                accs = []
                if frac == 0.0:
                    with torch.no_grad():
                        preds = forward_with_ablation(model, X, torch.ones(HIDDEN_SIZE))
                    accs.append(decision_accuracy(preds, Y))
                else:
                    n_ablate = int(round(frac * HIDDEN_SIZE))
                    for _ in range(N_REPEATS):
                        idx = rng.choice(HIDDEN_SIZE, n_ablate, replace=False)
                        mask = torch.ones(HIDDEN_SIZE)
                        mask[idx] = 0.0
                        with torch.no_grad():
                            preds = forward_with_ablation(model, X, mask)
                        accs.append(decision_accuracy(preds, Y))
                acc_mean = float(np.mean(accs))
                results.append((lam, seed, frac, acc_mean))
                print(f"lam={lam} seed={seed} frac={frac:.1f}  acc={acc_mean:.3f}")

    out_dir = Path(__file__).resolve().parent

    fig, ax = plt.subplots(figsize=(8, 5))
    for lam in LAMBDAS:
        means, sems = [], []
        for frac in ABLATION_FRACTIONS:
            accs = [a for l, _, f, a in results if l == lam and f == frac]
            means.append(np.mean(accs))
            sems.append(np.std(accs, ddof=1) / np.sqrt(len(accs)))
        ax.errorbar(ABLATION_FRACTIONS, means, yerr=sems, marker="o", capsize=3, label=f"λ={lam}")
    ax.axhline(0.5, color="k", lw=0.5, linestyle="--")
    ax.text(0.71, 0.51, "chance", fontsize=8, color="k")
    ax.set_xlabel("Fraction of hidden units ablated")
    ax.set_ylabel("Decision accuracy")
    ax.set_title("Robustness to random unit ablation across λ")
    ax.legend()
    fig.savefig(out_dir / "ablation_vs_lambda.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_dir / 'ablation_vs_lambda.png'}")

    with open(out_dir / "ablation_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lam", "seed", "ablation_fraction", "decision_accuracy"])
        for row in results:
            writer.writerow(row)
    print(f"saved -> {out_dir / 'ablation_results.csv'}")


if __name__ == "__main__":
    main()
