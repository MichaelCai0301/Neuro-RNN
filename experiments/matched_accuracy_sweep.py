"""Matched-accuracy continuation sweep.

Continues training each finetuned_lam{L}_seed{s}/epoch_030.pt until its val
decision accuracy clears 95% for two consecutive epochs (matched-accuracy
operating point) or a cumulative epoch cap of 150 is hit.

Saves the first checkpoint that meets the target as `target95.pt`, plus the
best-ever checkpoint as `best.pt`, and a `target_meta.npz` per run with the
cumulative epoch at which 95% was first achieved (NaN if never).

This is the matched-accuracy counterpart to the matched-effort sweep
(finetune_sweep.py), so downstream representational analyses (fidelity decoder,
participation ratio, RSA/Procrustes) can be run at a proficiency-controlled
operating point.
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tasks.poisson_dms import generate_trials
from rnn_model import VanillaRNN, TrialDataset, evaluate, extract_hidden_states

LAMBDAS = [2, 4, 6, 8, 10]
SEEDS = [0, 1, 2]

TARGET_ACC = 0.95
TARGET_CONSECUTIVE = 2          # must clear target for this many epochs in a row
CUMULATIVE_EPOCH_CAP = 150      # hard cap on total epochs from baseline
START_EPOCH = 30                # epoch of the checkpoint we're continuing from

NUM_TRAIN_TRIALS = 5000
NUM_VAL_TRIALS = 500
BATCH_SIZE = 64
LEARNING_RATE = 3e-4             # same preservation regime as the matched-effort sweep

DEVICE = 'cpu'
CK_ROOT = 'experiments/checkpoints'


def continue_one(lam, seed):
    tag = f"finetuned_lam{lam}_seed{seed}"
    run_dir = os.path.join(CK_ROOT, tag)
    resume_ck = os.path.join(run_dir, f'epoch_{START_EPOCH:03d}.pt')
    assert os.path.exists(resume_ck), f"missing {resume_ck}"

    train_seed = seed * 1000 + 42
    val_seed = seed * 1000 + 99
    X_train, Y_train, _ = generate_trials(num_trials=NUM_TRAIN_TRIALS, lam=lam, seed=train_seed)
    X_val, Y_val, _ = generate_trials(num_trials=NUM_VAL_TRIALS, lam=lam, seed=val_seed)

    class_counts = np.bincount(Y_train.flatten(), minlength=3)
    class_weights = torch.tensor(
        class_counts.sum() / (3 * class_counts), dtype=torch.float32,
    )
    train_loader = DataLoader(TrialDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TrialDataset(X_val, Y_val), batch_size=BATCH_SIZE, shuffle=False)

    # Restore model AND optimizer so the continuation is a true resume (Adam
    # first/second moment estimates preserved — otherwise we'd be restarting
    # momentum and the "same recipe" claim breaks).
    torch.manual_seed(seed)
    model = VanillaRNN(input_size=33, hidden_size=128, output_size=3).to(DEVICE)
    ck = torch.load(resume_ck, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck['model_state_dict'])

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    if 'optimizer_state_dict' in ck:
        optimizer.load_state_dict(ck['optimizer_state_dict'])
        # ensure lr matches what we want now (in case baseline used a different lr)
        for g in optimizer.param_groups:
            g['lr'] = LEARNING_RATE

    loss_function = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

    # --- Pull prior history so we produce a single end-to-end curve per run ---
    hist_path = os.path.join(run_dir, 'training_history.npz')
    prior = np.load(hist_path) if os.path.exists(hist_path) else None
    history = {k: list(prior[k]) if prior is not None else []
               for k in ['train_loss', 'train_acc', 'train_dec_acc',
                         'val_loss', 'val_acc', 'val_dec_acc']}

    target_epoch = None
    consecutive = 0
    best_val_dec = -1.0
    t0 = time.time()

    for epoch in range(START_EPOCH + 1, CUMULATIVE_EPOCH_CAP + 1):
        model.train()
        running_loss = 0.0; correct = 0; total = 0
        dec_correct = 0; dec_total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds, _ = model(xb)
            loss = loss_function(preds.reshape(-1, preds.size(-1)), yb.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            pc = preds.argmax(dim=-1)
            correct += (pc == yb).sum().item(); total += yb.numel()
            mask = yb >= 1
            if mask.any():
                dec_correct += (pc[mask] == yb[mask]).sum().item()
                dec_total += mask.sum().item()
        tr_loss = running_loss / len(train_loader.dataset)
        tr_acc = correct / total
        tr_dec = dec_correct / dec_total if dec_total else 0.0

        val_loss, val_acc, val_dec = evaluate(model, val_loader, loss_function, DEVICE)

        history['train_loss'].append(tr_loss); history['train_acc'].append(tr_acc)
        history['train_dec_acc'].append(tr_dec); history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc); history['val_dec_acc'].append(val_dec)

        print(f"[{tag}] Epoch {epoch:3d}/{CUMULATIVE_EPOCH_CAP} | "
              f"Loss {tr_loss:.4f} DecAcc {tr_dec:.4f} | "
              f"ValLoss {val_loss:.4f} ValDecAcc {val_dec:.4f}", flush=True)

        if val_dec > best_val_dec:
            best_val_dec = val_dec
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dec_acc': val_dec, 'train_dec_acc': tr_dec,
                'W_hh': model.W_rec.weight.detach().cpu().numpy(),
                'W_ih': model.W_in.weight.detach().cpu().numpy(),
            }, os.path.join(run_dir, 'best.pt'))

        if val_dec >= TARGET_ACC:
            consecutive += 1
        else:
            consecutive = 0

        if target_epoch is None and consecutive >= TARGET_CONSECUTIVE:
            target_epoch = epoch
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dec_acc': val_dec, 'train_dec_acc': tr_dec,
                'W_hh': model.W_rec.weight.detach().cpu().numpy(),
                'W_ih': model.W_in.weight.detach().cpu().numpy(),
            }, os.path.join(run_dir, 'target95.pt'))
            print(f"[{tag}] *** hit target 95% at cumulative epoch {epoch} ***", flush=True)
            break   # matched-accuracy: stop as soon as we hit target

    dt = time.time() - t0

    # Refresh history + hidden states at whichever checkpoint we'll use downstream
    np.savez(hist_path, **{k: np.array(v) for k, v in history.items()})
    np.savez(os.path.join(run_dir, 'target_meta.npz'),
             target_acc=TARGET_ACC, target_epoch=(target_epoch if target_epoch else np.nan),
             best_val_dec=best_val_dec, cap=CUMULATIVE_EPOCH_CAP)

    use_ck = 'target95.pt' if target_epoch else 'best.pt'
    ck_for_h = torch.load(os.path.join(run_dir, use_ck), map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck_for_h['model_state_dict'])
    H = extract_hidden_states(model, X_val[:50], device=DEVICE)
    np.save(os.path.join(run_dir, 'hidden_states_val_matched.npy'), H)

    msg = (f"[{tag}] done in {dt/60:.1f} min | "
           f"target@{target_epoch if target_epoch else 'NEVER'} | best={best_val_dec:.4f}")
    print(msg, flush=True)
    return {'lam': lam, 'seed': seed, 'target_epoch': target_epoch,
            'best_val_dec': best_val_dec, 'wall_s': dt}


def main():
    t_all = time.time()
    results = []
    for lam in LAMBDAS:
        for seed in SEEDS:
            results.append(continue_one(lam, seed))

    print(f"\n{'='*60}\nMATCHED-ACCURACY SUMMARY (total {(time.time()-t_all)/60:.1f} min)\n{'='*60}")
    print(f"{'lam':>5} {'seed':>5} {'TargetEpoch':>12} {'Best':>8} {'Wall(s)':>9}")
    for r in results:
        te = r['target_epoch'] if r['target_epoch'] else '—'
        print(f"{r['lam']:>5} {r['seed']:>5} {str(te):>12} {r['best_val_dec']:>8.4f} {r['wall_s']:>9.1f}")
    print(f"\n{'lam':>5} {'hit':>5} {'mean_epoch_to_95':>18} {'mean_best':>10}")
    for lam in LAMBDAS:
        rs = [r for r in results if r['lam'] == lam]
        hits = [r['target_epoch'] for r in rs if r['target_epoch'] is not None]
        mean_ep = f"{np.mean(hits):.1f}" if hits else '—'
        print(f"{lam:>5} {len(hits):>3}/3 {mean_ep:>18} {np.mean([r['best_val_dec'] for r in rs]):>10.4f}")


if __name__ == '__main__':
    main()
