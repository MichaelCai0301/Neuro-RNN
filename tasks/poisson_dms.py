"""
Delayed Match-to-Sample with Poisson distractors during the delay period.

No neurogym dependency — generates (obs, labels, metadata) tensors directly.

Trial structure (all durations in ms, dt=20ms):
    fixation (300ms, 15 steps): fixation signal ON, no stimulus
    sample   (500ms, 25 steps): target angle presented as cosine bump on ring
    delay   (2500ms, 125 steps): stimulus off, Poisson distractors injected
    test     (500ms, 25 steps): test angle presented
    decision (500ms, 25 steps): fixation OFF, network must output match/non-match

Observation space: 33 dims (1 fixation + 32 ring channels)
Labels: 0 = fixation, 1 = match, 2 = non-match
"""

import numpy as np


# Timing in ms
TIMING = {
    'fixation': 300,
    'sample': 500,
    'delay': 2500,
    'test': 500,
    'decision': 500,
}
DT = 20  # ms per timestep
DIM_RING = 32  # number of orientation-tuned channels
DISTRACTOR_SLOT_MS = 200  # ms per distractor slot in the delay
EXCLUSION_WINDOW = 0.2  # radians — min distance between distractor/test and target


def _cosine_bump(angle, dim_ring=DIM_RING):
    """Cosine tuning curve centered at `angle`, matching NeuroGym's encoding."""
    preferred = np.linspace(0, 2 * np.pi, dim_ring, endpoint=False)
    return np.cos(preferred - angle) * 0.5 + 0.5


def _angular_distance(a, b):
    """Unsigned circular distance between two angles."""
    d = np.abs(a - b) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)


def _sample_away_from(target, exclusion, rng):
    """Sample an angle uniformly from [0, 2pi) but at least `exclusion` rad from target."""
    while True:
        angle = rng.uniform(0, 2 * np.pi)
        if _angular_distance(angle, target) >= exclusion:
            return angle


def generate_trial(lam=0.0, rng=None):
    """Generate a single trial.

    Args:
        lam: Poisson rate for number of distractors in the delay period.
        rng: numpy RandomState for reproducibility.

    Returns:
        obs: (T, 33) float array — network input
        label: (T,) int array — ground truth per timestep
        info: dict with trial metadata
    """
    if rng is None:
        rng = np.random.RandomState()

    # Compute timesteps per period
    steps = {k: int(v / DT) for k, v in TIMING.items()}
    total_steps = sum(steps.values())

    # Build observation tensor: 1 fixation + 32 ring channels
    obs = np.zeros((total_steps, 1 + DIM_RING), dtype=np.float32)
    label = np.zeros(total_steps, dtype=np.int64)

    # Period boundaries (start, end) in timestep indices
    t = 0
    boundaries = {}
    for period in ['fixation', 'sample', 'delay', 'test', 'decision']:
        boundaries[period] = (t, t + steps[period])
        t += steps[period]

    # --- Fixation signal: ON during fixation/sample/delay/test, OFF during decision ---
    fix_end = boundaries['decision'][0]
    obs[:fix_end, 0] = 1.0

    # --- Sample stimulus ---
    target_angle = rng.uniform(0, 2 * np.pi)
    bump = _cosine_bump(target_angle)
    s, e = boundaries['sample']
    obs[s:e, 1:] = bump[np.newaxis, :]

    # --- Distractors during delay ---
    n_distractors = rng.poisson(lam)
    delay_s, delay_e = boundaries['delay']
    delay_steps = steps['delay']
    slot_steps = int(DISTRACTOR_SLOT_MS / DT)  # 10 steps per slot
    n_slots = delay_steps // slot_steps  # 12 slots in 2500ms delay
    n_distractors = min(n_distractors, n_slots)  # cap at available slots

    distractor_angles = []
    distractor_onsets = []  # in absolute timestep indices

    if n_distractors > 0:
        # Sample slot positions without replacement
        chosen_slots = rng.choice(n_slots, size=n_distractors, replace=False)
        chosen_slots.sort()

        for slot_idx in chosen_slots:
            d_angle = _sample_away_from(target_angle, EXCLUSION_WINDOW, rng)
            d_onset = delay_s + slot_idx * slot_steps
            d_offset = min(d_onset + slot_steps, delay_e)

            d_bump = _cosine_bump(d_angle)
            obs[d_onset:d_offset, 1:] = d_bump[np.newaxis, :]

            distractor_angles.append(d_angle)
            distractor_onsets.append(d_onset)

    # --- Test stimulus ---
    is_match = rng.rand() < 0.5
    if is_match:
        test_angle = target_angle
    else:
        test_angle = _sample_away_from(target_angle, EXCLUSION_WINDOW, rng)

    s, e = boundaries['test']
    test_bump = _cosine_bump(test_angle)
    obs[s:e, 1:] = test_bump[np.newaxis, :]

    # --- Add input noise (during stimulus periods only) ---
    noise_sigma = 0.1
    for period in ['sample', 'test']:
        s, e = boundaries[period]
        obs[s:e, 1:] += rng.randn(e - s, DIM_RING).astype(np.float32) * noise_sigma
    # Also add noise during distractor windows
    for d_onset in distractor_onsets:
        d_offset = min(d_onset + slot_steps, delay_e)
        obs[d_onset:d_offset, 1:] += rng.randn(d_offset - d_onset, DIM_RING).astype(np.float32) * noise_sigma

    # --- Labels ---
    # 0 everywhere except decision period
    s, e = boundaries['decision']
    label[s:e] = 1 if is_match else 2

    info = {
        'target_angle': target_angle,
        'test_angle': test_angle,
        'is_match': is_match,
        'n_distractors': n_distractors,
        'distractor_angles': np.array(distractor_angles),
        'distractor_onsets': np.array(distractor_onsets),
        'boundaries': boundaries,
    }

    return obs, label, info


def generate_trials(num_trials=1000, lam=0.0, seed=42):
    """Generate a batch of trials.

    Returns:
        X: (num_trials, T, 33) observations
        Y: (num_trials, T) labels
        infos: list of dicts with per-trial metadata
    """
    rng = np.random.RandomState(seed)

    all_obs, all_labels, all_infos = [], [], []
    for _ in range(num_trials):
        obs, label, info = generate_trial(lam=lam, rng=rng)
        all_obs.append(obs)
        all_labels.append(label)
        all_infos.append(info)

    X = np.stack(all_obs)
    Y = np.stack(all_labels)
    return X, Y, all_infos
