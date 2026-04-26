"""
Training sweep: train networks for every (lambda, seed) combination.
lambda in {0, 0.5, 1, 2, 4}, seed in {0, 1, 2} = 15 networks total.
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tasks.poisson_dms import generate_trials
from rnn_model import VanillaRNN, TrialDataset, train_model, extract_hidden_states

# Sweep parameters
LAMBDAS = [0, 0.5, 1, 2, 4]
SEEDS = [0, 1, 2]
NUM_TRAIN_TRIALS = 5000
NUM_VAL_TRIALS = 500
BATCH_SIZE = 64
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 128
CHECKPOINT_EVERY = 50  # save less frequently to reduce disk usage

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

results = []

for lam in LAMBDAS:
    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"Training: lambda={lam}, seed={seed}")
        print(f"{'='*60}")

        # Use different seeds for train/val data generation per run
        train_seed = seed * 1000 + 42
        val_seed = seed * 1000 + 99

        X_train, Y_train, _ = generate_trials(
            num_trials=NUM_TRAIN_TRIALS, lam=lam, seed=train_seed)
        X_val, Y_val, _ = generate_trials(
            num_trials=NUM_VAL_TRIALS, lam=lam, seed=val_seed)

        num_classes = len(np.unique(Y_train))
        class_counts = np.bincount(Y_train.flatten(), minlength=num_classes)
        class_weights = torch.tensor(
            class_counts.sum() / (num_classes * class_counts),
            dtype=torch.float32)

        train_loader = DataLoader(
            TrialDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(
            TrialDataset(X_val, Y_val), batch_size=BATCH_SIZE, shuffle=False)

        # Set torch seed for weight initialization reproducibility
        torch.manual_seed(seed)
        model = VanillaRNN(
            input_size=33, hidden_size=HIDDEN_SIZE, output_size=num_classes)

        checkpoint_dir = os.path.join(
            'experiments', 'checkpoints', f'lam{lam}_seed{seed}')
        history = train_model(
            model, train_loader, val_loader=val_loader,
            num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE,
            checkpoint_dir=checkpoint_dir, checkpoint_every=CHECKPOINT_EVERY,
            device=DEVICE, class_weights=class_weights)

        # Save training history
        np.savez(
            os.path.join(checkpoint_dir, 'training_history.npz'),
            train_loss=history['train_loss'], train_acc=history['train_acc'],
            train_dec_acc=history['train_dec_acc'], val_loss=history['val_loss'],
            val_acc=history['val_acc'], val_dec_acc=history['val_dec_acc'])

        # Extract hidden states for analysis
        hidden_states = extract_hidden_states(model, X_val[:50], device=DEVICE)
        np.save(os.path.join(checkpoint_dir, 'hidden_states_val.npy'), hidden_states)

        final_train_dec = history['train_dec_acc'][-1]
        final_val_dec = history['val_dec_acc'][-1]
        results.append({
            'lam': lam, 'seed': seed,
            'train_dec_acc': final_train_dec,
            'val_dec_acc': final_val_dec,
        })

        print(f"\n  Final: TrainDecAcc={final_train_dec:.4f}, ValDecAcc={final_val_dec:.4f}")

# Print summary table
print(f"\n\n{'='*60}")
print("SWEEP SUMMARY")
print(f"{'='*60}")
print(f"{'Lambda':>8} {'Seed':>6} {'TrainDecAcc':>12} {'ValDecAcc':>12} {'Status':>10}")
print("-" * 52)
for r in results:
    status = "OK" if r['val_dec_acc'] > 0.70 else "FAIL"
    print(f"{r['lam']:>8} {r['seed']:>6} {r['train_dec_acc']:>12.4f} {r['val_dec_acc']:>12.4f} {status:>10}")

# Summary by lambda
print(f"\n{'Lambda':>8} {'Mean ValDecAcc':>16} {'Std':>8}")
print("-" * 36)
for lam in LAMBDAS:
    accs = [r['val_dec_acc'] for r in results if r['lam'] == lam]
    print(f"{lam:>8} {np.mean(accs):>16.4f} {np.std(accs):>8.4f}")
