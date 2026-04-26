

import os
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tasks.poisson_dms import generate_trials
from rnn_model import (
    VanillaRNN, TrialDataset, train_model, extract_hidden_states,
)

NEW_LAMBDAS = [2, 4, 6, 8, 10]
SEEDS = [0, 1, 2]

NUM_TRAIN_TRIALS = 5000
NUM_VAL_TRIALS = 500
BATCH_SIZE = 64
NUM_EPOCHS = 30
LEARNING_RATE = 3e-4
HIDDEN_SIZE = 128
CHECKPOINT_EVERY = 10

DEVICE = 'cpu' 
BASELINE_DIR = 'experiments/checkpoints'
OUT_DIR = 'experiments/checkpoints'


def run_one(lam, seed):
    tag = f"finetuned_lam{lam}_seed{seed}"
    out_dir = os.path.join(OUT_DIR, tag)
    baseline_ck = os.path.join(BASELINE_DIR, f'lam0_seed{seed}', 'epoch_100.pt')
    assert os.path.exists(baseline_ck), f"missing baseline: {baseline_ck}"

    train_seed = seed * 1000 + 42
    val_seed = seed * 1000 + 99

    X_train, Y_train, _ = generate_trials(num_trials=NUM_TRAIN_TRIALS, lam=lam, seed=train_seed)
    X_val, Y_val, _ = generate_trials(num_trials=NUM_VAL_TRIALS, lam=lam, seed=val_seed)

    num_classes = 3
    class_counts = np.bincount(Y_train.flatten(), minlength=num_classes)
    class_weights = torch.tensor(
        class_counts.sum() / (num_classes * class_counts), dtype=torch.float32,
    )

    train_loader = DataLoader(TrialDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TrialDataset(X_val, Y_val), batch_size=BATCH_SIZE, shuffle=False)

    torch.manual_seed(seed)
    model = VanillaRNN(input_size=33, hidden_size=HIDDEN_SIZE, output_size=num_classes)

    baseline = torch.load(baseline_ck, map_location='cpu', weights_only=False)
    model.load_state_dict(baseline['model_state_dict'])
    print(f"[{tag}] loaded baseline val_dec_acc≈ (from lam0_seed{seed} ep100)", flush=True)

    t0 = time.time()
    history = train_model(
        model, train_loader, val_loader=val_loader,
        num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE,
        checkpoint_dir=out_dir, checkpoint_every=CHECKPOINT_EVERY,
        device=DEVICE, class_weights=class_weights, log_prefix=f"[{tag}] ",
    )
    dt = time.time() - t0

    np.savez(
        os.path.join(out_dir, 'training_history.npz'),
        train_loss=history['train_loss'], train_acc=history['train_acc'],
        train_dec_acc=history['train_dec_acc'], val_loss=history['val_loss'],
        val_acc=history['val_acc'], val_dec_acc=history['val_dec_acc'],
    )

    hidden_states = extract_hidden_states(model, X_val[:50], device=DEVICE)
    np.save(os.path.join(out_dir, 'hidden_states_val.npy'), hidden_states)

    print(f"[{tag}] done in {dt/60:.1f} min | final ValDecAcc={history['val_dec_acc'][-1]:.4f}", flush=True)
    return {
        'lam': lam, 'seed': seed,
        'train_dec_acc': history['train_dec_acc'][-1],
        'val_dec_acc': history['val_dec_acc'][-1],
        'wall_s': dt,
    }


def main():
    t_all = time.time()
    results = []
    for lam in NEW_LAMBDAS:
        for seed in SEEDS:
            results.append(run_one(lam, seed))

    print(f"\n{'='*60}\nFINE-TUNE SWEEP SUMMARY (total {(time.time()-t_all)/60:.1f} min)\n{'='*60}")
    print(f"{'lam':>5} {'seed':>5} {'TrainDec':>10} {'ValDec':>10} {'Wall(s)':>9}")
    for r in results:
        print(f"{r['lam']:>5} {r['seed']:>5} {r['train_dec_acc']:>10.4f} {r['val_dec_acc']:>10.4f} {r['wall_s']:>9.1f}")
    print(f"\n{'lam':>5} {'mean_val':>10} {'std':>8}")
    for lam in NEW_LAMBDAS:
        accs = [r['val_dec_acc'] for r in results if r['lam'] == lam]
        print(f"{lam:>5} {np.mean(accs):>10.4f} {np.std(accs):>8.4f}")


if __name__ == '__main__':
    main()
