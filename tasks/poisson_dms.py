"""
HELPER MODULE:

Delayed Match-to-Sample (DMS) with Poisson distributed distractors during the delay period.
(note: we based a lot of this on neurogym but ended up not using the library since it made things much more
complicated customizability-wise).

INPUTS (data): each input is a trial (described below)
Trial structure (all durations in ms, dt=20ms for default case):
    fixation (300ms, 15 steps): fixation signal ON, no stimulus
    sample (500ms, 25 steps): target angle presented as cosine bump on ring
    delay (2500ms, 125 steps): stimulus off, Poisson distractors injected
    test (500ms, 25 steps): test angle presented. the model needs to determine if the test angle matches the sample target angle
    decision (500ms, 25 steps): fixation OFF, network must output match/non-match

Labels: 0 = fixation, 1 = match, 2 = non-match


(Note: a lot of this was mathematically/theoretically intensive to get right so we used AI to help debug some functions,
but the foundation and vast majority of final code was written and debugged only by us)

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
DT = 20 # ms per timestep (dt)
DIM_NEURON = 32 # number of orientation-tuned channels (1 per neuron)
DISTRACTOR_SLOT_MS = 200 # ms per distractor slot in the delay
EXCLUSION_WINDOW = 0.2 # radians — min distance between distractor/test and target


def cosine_bump(angle, dim_neuron=DIM_NEURON):
    """
    This function creates a cosine tuning curve that has center `angle` (we got this inspiration from the NeuroGym encoding)
    Each neuron's preferred angle is an angle that is evenly spaced from all the other neuron's preferred angles (that's how we get `preferred`)
    """
    preferred = np.linspace(0, 2 * np.pi, dim_neuron, endpoint=False)
    return np.cos(preferred - angle) * 0.5 + 0.5


def angular_distance(a, b):
    """
    unsigned circular distance between two angles
    """
    d = np.abs(a - b) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)


def sample_away_from(target, exclusion, rng):
    """
    Sample an angle uniformly from [0, 2pi) but at least `exclusion` rad from target
    (I'm not sure if this is the most efficient way to do this tho)
    """
    while True:
        angle = rng.uniform(0, 2 * np.pi)
        if angular_distance(angle, target) >= exclusion:
            return angle


def generate_trial(lam=0.0, rng=None):
    """
    Generate a single trial: lam = Poisson rate for number of distractors in the delay period, rng = numpy RandomState for reproducibility
    Returns:
    - obs: (T, 33) network input
    - label: (T,) ground truth per timestep
    - info: dict with trial metadata
    """
    if rng is None:
        rng = np.random.RandomState()
    steps = {k: int(v / DT) for k, v in TIMING.items()} # quick wway of making time step dict
    total_steps = sum(steps.values())

    # Build observation tensor: 1 fixation + 32 channels
    # KEEP THE DATATYPES SPECS! sometiems fails otherwise
    obs = np.zeros((total_steps, 1 +DIM_NEURON), dtype=np.float32)
    label = np.zeros(total_steps, dtype=np.int64)

    t = 0
    boundaries = {}
    for period in ['fixation', 'sample', 'delay', 'test', 'decision']:
        boundaries[period] = (t, t+steps[period])
        t += steps[period]

    # Fixation signal
    fix_end = boundaries['decision'][0]
    obs[:fix_end, 0] = 1.0

    # Sample stimulus
    target_angle = rng.uniform(0, 2 * np.pi)
    bump = cosine_bump(target_angle)
    s,e = boundaries['sample']
    obs[s:e, 1:] = bump[np.newaxis, :]

    # Distractors during delay
    n_distractors = rng.poisson(lam)
    delay_s, delay_e = boundaries['delay']
    delay_steps = steps['delay']
    slot_steps = int(DISTRACTOR_SLOT_MS/DT) # 10 steps per slot
    n_slots = delay_steps//slot_steps # 12 slots in 2500ms(here) delay
    n_distractors = min(n_distractors, n_slots) # cap at available slots

    distractor_angles = []
    distractor_onsets = []

    if n_distractors > 0:
        # Sample slot positions w/o replacement
        chosen_slots = rng.choice(n_slots, size=n_distractors, replace=False)
        chosen_slots.sort()
        for slot_idx in chosen_slots:
            d_angle = sample_away_from(target_angle, EXCLUSION_WINDOW, rng)
            d_onset = delay_s + slot_idx * slot_steps
            d_offset = min(d_onset + slot_steps, delay_e)
            d_bump = cosine_bump(d_angle)
            obs[d_onset:d_offset, 1:] = d_bump[np.newaxis, :]
            distractor_angles.append(d_angle)
            distractor_onsets.append(d_onset)

    # Test stimulus
    is_match = rng.rand() < 0.5
    if is_match:
        test_angle = target_angle
    else:
        test_angle = sample_away_from(target_angle, EXCLUSION_WINDOW, rng)

    s, e = boundaries['test']
    test_bump = cosine_bump(test_angle)
    obs[s:e, 1:] = test_bump[np.newaxis,:]

    # Add input noise (during stimulus periods only) -- this is done in the NeuroGym implementation and is biologically reasonable, so we
    # include it here
    noise_sigma = 0.1
    for period in ['sample', 'test']:
        s, e = boundaries[period]
        obs[s:e, 1:] += rng.randn(e - s, DIM_NEURON).astype(np.float32) * noise_sigma
    # Also add noise during distractor windows (like before)
    for d_onset in distractor_onsets:
        d_offset = min(d_onset + slot_steps, delay_e)
        obs[d_onset:d_offset, 1:] += rng.randn(d_offset - d_onset, DIM_NEURON).astype(np.float32) * noise_sigma

    # Labels, 0 everywhere except decision period
    s, e = boundaries['decision']
    label[s: e] = 1 if is_match else 2

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
    """
    Generate a batch of trials.
    Returns:
    - X: (num_trials, T, 33) observations
    - Y: (num_trials, T) labels
    - infos: list of dicts with per-trial metadata
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
