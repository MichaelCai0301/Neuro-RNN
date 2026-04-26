# Neuro-RNN

Working-memory RNNs trained on a Delayed Match-to-Sample (DMS) task with Poisson-distributed distractors injected during the delay period. The structural hypothesis: networks trained under stronger distraction develop more highly structured connectivity and lower-dimensional dynamics than networks trained without distraction.

## Repo layout

- `rnn_model.py` — `VanillaRNN` (continuous-time leaky-integrator RNN, τ=500 ms, 33 → 128 → 3) and the training loop.
- `tasks/poisson_dms.py` — custom DMS task with Poisson-distributed distractors on a 32-channel ring encoding.
- `experiments/run_sweep.py` — Phase 1: from-scratch baselines (λ ∈ {0, 0.5, 1, 2, 4} × 3 seeds, 100 epochs, lr=1e-3).
- `experiments/finetune_sweep.py` — Phase 2: matched-effort fine-tunes from each λ=0 baseline to λ ∈ {2, 4, 6, 8, 10} × 3 seeds, 30 epochs at lr=3e-4.
- `experiments/matched_accuracy_sweep.py` — Phase 3: continues each Phase-2 network until val_dec_acc ≥ 95% for two consecutive epochs (cap 150).
- `experiments/checkpoints/` — all trained model checkpoints.

## The 18 networks for analysis

3 baselines + 15 fine-tunes, all sharing the same `VanillaRNN(33, 128, 3)` architecture:

```
experiments/checkpoints/
├── lam0_seed{0,1,2}/                   # Phase 1 baselines (clean delay, ≥99% val dec acc)
│   └── epoch_100.pt
└── finetuned_lam{L}_seed{s}/           # Phase 2+3, L ∈ {2,4,6,8,10}, s ∈ {0,1,2}
    ├── epoch_{010,020,030}.pt          # Phase 2 checkpoints (matched-effort)
    ├── target95.pt                     # ← matched-accuracy checkpoint — use this for analysis
    ├── best.pt                         # highest val_dec_acc ever seen (safety fallback)
    ├── target_meta.npz                 # target_acc, target_epoch, best_val_dec, cap
    ├── training_history.npz            # end-to-end loss/acc curves across Phase 2+3
    ├── hidden_states_val.npy           # 50 val trials × 215 × 128 at epoch_030.pt
    └── hidden_states_val_matched.npy   # 50 val trials × 215 × 128 at target95.pt
```

Lineage: each `finetuned_lam{L}_seed{s}` is forked from the matching-seed baseline `lam0_seed{s}/epoch_100.pt`, then trained at the new λ. Seeds are matched, so all five λ forks within a seed share a common ancestor — representational differences across λ within a seed are attributable to distractor exposure, not to landing in arbitrarily different solution basins from independent random inits.

## Training pipeline

| phase | what varies | held constant | what it produces | what it removes |
|---|---|---|---|---|
| 1 — from-scratch baselines | λ, seed | 100 ep, lr=1e-3, fresh init | λ=0 anchor checkpoints (≥99% val dec acc) | "is the task even learnable?" |
| 2 — matched-effort fine-tune | λ, seed | 30 ep, lr=3e-4, **baseline init** | accuracy-vs-λ under fixed budget; networks stay near baseline basin | solution-rotation noise across seeds/λ |
| 3 — matched-accuracy continuation | training length | lr=3e-4, baseline lineage, target val_dec_acc=95% for 2 consecutive epochs | all 15 networks at the same behavioral operating point | proficiency confound |

Phase 2 results (final-epoch val dec acc, mean ± std over 3 seeds): 96.7% ± 0.4 (λ=2), 96.5% ± 0.5 (λ=4), 95.2% ± 1.2 (λ=6), 91.6% ± 1.1 (λ=8), 91.0% ± 2.3 (λ=10).

Phase 3 results: all 15 hit the 95% target inside the cap. λ ∈ {2, 4, 6} hit it almost immediately (~32 cumulative epochs from baseline); λ ∈ {8, 10} need ~2× longer (~61–65 epochs) — a standalone "convergence-speed cost of distractor load" finding alongside the accuracy-ceiling story.

## Loading a checkpoint

```python
import torch
from rnn_model import VanillaRNN

model = VanillaRNN(input_size=33, hidden_size=128, output_size=3)
ck = torch.load('experiments/checkpoints/finetuned_lam8_seed0/target95.pt',
                map_location='cpu', weights_only=False)
model.load_state_dict(ck['model_state_dict'])
model.eval()
```

For weight-only analyses (SVD of `W_rec`, weight-distance to baseline, etc.) skip the model:

```python
W_hh = ck['W_hh']   # (128, 128) recurrent weights
W_ih = ck['W_ih']   # (128, 33) input weights
```

## Regenerating trial metadata

Per-trial metadata (`target_angle`, `distractor_onsets`, `is_match`, `n_distractors`, `distractor_angles`) is **not** saved alongside the hidden states — regenerate it deterministically from the seed convention used by every sweep script:

| split | seed formula |
|---|---|
| train | `seed * 1000 + 42` |
| val   | `seed * 1000 + 99` |

```python
import numpy as np
from tasks.poisson_dms import generate_trials

# Aligns with hidden_states_val_matched.npy for run (lam=L, seed=s):
X, Y, infos = generate_trials(num_trials=500, lam=L, seed=s * 1000 + 99)
infos_aligned = infos[:50]
target_angles = np.array([info['target_angle'] for info in infos_aligned])
is_match      = np.array([info['is_match']     for info in infos_aligned])
n_distractors = np.array([info['n_distractors'] for info in infos_aligned])
```

For the λ=0 baselines use `lam=0` and the same seed formula. Match is exact — `generate_trials` only consumes its own `np.random.RandomState(seed)`.

## Trial timing (timesteps, dt=20 ms)

| period   | start | end | length |
|----------|------:|----:|-------:|
| fixation |     0 |  15 |     15 |
| sample   |    15 |  40 |     25 |
| delay    |    40 | 165 |    125 |
| test     |   165 | 190 |     25 |
| decision |   190 | 215 |     25 |

Total: 215 timesteps per trial. Source of truth: `tasks/poisson_dms.TIMING` and `DT`.

## Which file feeds which analysis

| analysis | inputs |
|---|---|
| Fidelity decoder (ridge: hidden → [sin θ, cos θ]) | `hidden_states_val_matched.npy` + regenerated `target_angles` |
| Participation ratio of delay-period activity | `hidden_states_val_matched.npy[:, 40:165, :]` |
| PCA(hidden) ↔ SVD(W_rec) alignment | `hidden_states_val_matched.npy` + `ck['W_hh']` |
| Cross-network comparisons across λ | `hidden_states_val_matched.npy` from each of the 15 fine-tunes (+ baseline `hidden_states_val.npy` from `lam0_seed{s}` for reference) |
| Baseline-to-matched weight drift vs λ | `lam0_seed{s}/epoch_100.pt['W_hh']` vs `finetuned_lam{L}_seed{s}/target95.pt['W_hh']` |

No GPU required — checkpoints were trained on CPU.
