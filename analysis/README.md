# Analysis pipeline

This folder contains the four analyses we run on the trained networks to test our hypothesis. Each one is a standalone Python script you can run from the repo root.

```bash
python analysis/testPR.py          # Step 1
python analysis/testSpectrum.py    # Step 1 (supporting)
python analysis/testSVD.py         # Step 2
python analysis/testAblation.py    # Step 3
```

## Big picture: what are we trying to learn?

We trained 18 networks total: 3 baselines with no distractors (λ=0) and 15 networks at progressively stronger distractor levels (λ ∈ {2, 4, 6, 8, 10}, 3 random seeds each). All 18 networks reach ≥95% accuracy on the working-memory task. They behave the same on the *outside*.

The question is: do they look the same on the *inside*?

Our hypothesis was: **networks trained with more distractors should develop more structured, lower-dimensional representations** — i.e., the way they use their 128 hidden neurons should become more compact and stereotyped as λ goes up.

The four analyses test this from three different angles:

| Step | What it looks at | Plain-English question |
|------|------------------|------------------------|
| 1. PCA → Participation Ratio | Hidden-state activity during the delay | How many "axes" of activity does the network use to hold the memory? |
| 1b. PCA → Eigenvalue spectrum | Same | Same question, but showing the full curve instead of one summary number. |
| 2. SVD on `W_rec` | The recurrent weight matrix | Is the network's *wiring* concentrated into a few dominant patterns, or spread out? |
| 3. Random unit ablation | Decision accuracy after killing some neurons | If we remove some hidden units, does the network still work? Which networks are more robust? |

**A quick note on terminology:** PCA and Participation Ratio aren't the same thing. PCA is the *procedure* we covered in class (the six steps: standardize → covariance → eigenvectors → choose basis → project → analyze). Participation Ratio is one specific way to *summarize* the output of PCA — it takes the eigenvalues PCA produces in step 3 and collapses them into a single "effective number of dimensions" scalar via the formula $\mathrm{PR} = (\sum_i \lambda_i)^2 / \sum_i \lambda_i^2$. So PR is built on top of PCA, not separate from it. Steps 1 and 1b both run PCA on the same data; they just summarize the result differently (one number vs. the full curve).

---

## Data setup (what each script reads from disk)

All checkpoints live in `experiments/checkpoints/`. Every λ has 3 seeds. Two kinds of files matter for analysis:

| File | Shape | What it is | Used by |
|------|-------|-----------|---------|
| `lam0_seed{s}/hidden_states_val.npy` | `(50, 215, 128)` | Hidden activity, λ=0 baseline | Steps 1, 1b |
| `finetuned_lam{L}_seed{s}/hidden_states_val_matched.npy` | `(50, 215, 128)` | Hidden activity at matched-accuracy checkpoint, λ ∈ {2,4,6,8,10} | Steps 1, 1b |
| `lam0_seed{s}/epoch_100.pt` | — | Trained baseline checkpoint. Contains `W_hh` (recurrent weight matrix) and full state dict. **Also serves as W₀ — the initialization that each finetuned network was forked from** | Steps 2, 3 |
| `finetuned_lam{L}_seed{s}/target95.pt` | — | Trained matched-accuracy checkpoint, contains `W_hh` and full state dict | Steps 2, 3 |

Hidden-state arrays are `(n_trials=50, T=215 timesteps, H=128 units)`. Within the 215 timesteps: fixation 0–14, sample 15–39, **delay 40–164**, test 165–189, decision 190–214. The delay window is what every PCA-style analysis slices into.

Each PCA-based analysis (Steps 1, 1b) loops over all 18 networks (6 λ × 3 seeds). Step 2 (SVD) runs on the 15 finetuned networks only — λ=0 has no comparable perturbation since the baseline IS the reference. Step 3 (ablation) runs on all 18. All scripts aggregate within each λ across the 3 seeds (mean ± SEM where applicable).

---

## Step 1 — `testPR.py`: PCA, summarized as Participation Ratio

### How this relates to the six-step PCA procedure

The six steps from class are: (1) standardize the data matrix, (2) compute the covariance matrix, (3) find the eigenvectors, (4) define a new basis with some subset of those eigenvectors, (5) project data onto the new basis, (6) analyze the results. We do steps 1–3 in full. Steps 4–5 (basis selection and projection) we skip, because they're for visualizing data in a lower-dimensional space — that's not the question we're asking. Instead, step 6 ("analyze the results") is where Participation Ratio comes in: PR is a one-number summary of the eigenvalues PCA produces, designed to estimate *how many dimensions are actually being used*. So we're running PCA exactly as taught, and then summarizing its output with PR rather than projecting and plotting.

### The intuition

Every trial, the 128 hidden neurons produce a long sequence of activity values during the delay period (when the network is supposed to be holding the memory). You can imagine each timestep as a single point in a 128-dimensional space, where each axis is one neuron.

If only a handful of neurons are doing the work, those 128-D points all lie on a 2D or 3D plane embedded in the 128-D space — the "real" dimensionality is small. If every neuron is contributing roughly equally, the points fill all 128 dimensions evenly.

PCA tells us the directions in 128-D space along which the activity varies most. The eigenvalues that come out of PCA tell us *how much variance* each direction carries. The **Participation Ratio (PR)** is one number that summarizes the eigenvalue list:

$$\mathrm{PR} = \frac{(\sum_i \lambda_i)^2}{\sum_i \lambda_i^2}$$

If one direction dominates: PR ≈ 1. If k directions are equally important: PR ≈ k. So PR is the "effective number of dimensions" the activity actually uses.

### What we compute (per network)

**Reads:** `hidden_states_val_matched.npy` (or `hidden_states_val.npy` for λ=0) — one array per network.
**Writes:** `pr_vs_lambda.png`, `pr_results.csv`.

1. Slice the hidden states down to the delay window only (timesteps 40–165).
2. Center each neuron's activity (subtract its mean).
3. Compute the 128×128 covariance matrix.
4. Get its eigenvalues.
5. Plug them into the PR formula.

That gives one PR scalar per network. After looping all 18 networks, we group by λ and average the 3 seeds' PRs (with SEM error bars).

### What the plot shows

`pr_vs_lambda.png` plots PR vs λ, averaged over the 3 seeds with error bars.

PR rises from **~3.2 at λ=0 to ~4.4 at λ=10**. The endpoint error bars don't overlap, so the trend is real. The middle λ values are roughly flat within seed variance.

### What this means

**Higher distraction → networks use *more* dimensions, not fewer.** This is the opposite of what we hypothesized. One way to read this: when you train a network in a noisy environment, it has to maintain extra "axes" to distinguish the real signal from distractors. Collapsing onto a single memory axis (which would give PR=1) would be too fragile — the network needs spare capacity to disambiguate.

The numbers are also extremely small overall (3–5 out of a possible 128). That's consistent with the broader working-memory literature, which has long known that persistent activity lives in very low-dimensional subspaces. We're just seeing how that low-dimensionality shifts with training condition.

---

## Step 1b — `testSpectrum.py`: PCA, summarized as the eigenvalue spectrum

### Why a second look at the same PCA?

This script runs the *same* PCA as Step 1 — same centering, same covariance matrix, same eigendecomposition. The only difference is how we summarize the result. PR collapses the eigenvalue list into one number; the spectrum plot keeps the full list and shows it as a curve. Two networks with the same PR can still have different *shapes* of variance distribution, so the spectrum is a sanity check that the PR result isn't an artifact of how we compressed it into a scalar.

### What we compute

**Reads:** same hidden-state files as Step 1.
**Writes:** `spectrum_vs_lambda.png` (cumulative variance), `eigenvalues_vs_lambda.png` (raw eigenvalues, log scale).

For each network, the same 128 eigenvalues from PCA, sorted big-to-small. Then we compute cumulative variance explained: how much total variance you capture if you keep just the top-N eigenvalues. The 3 seeds within each λ are averaged, then we plot one curve per λ.

### What the plot shows

`spectrum_vs_lambda.png`: at λ=0, the top PC alone captures ~50% of the variance, and PC1–4 together capture ~90%. At λ=10, PC1 captures only ~38%, and you need ~7 PCs to reach 90%.

`eigenvalues_vs_lambda.png`: the raw spectrum on a log scale. λ=0 has a steeper drop-off; high-λ has a shallower decay.

### What this means

The PR result isn't a quirky artifact. The variance is genuinely shifting from PC1 toward PC2–5 as λ increases. **The dynamics get less concentrated in any single direction** — a real, cross-validated finding.

---

## Step 2 — `testSVD.py`: SVD on the learning-induced weight perturbation

### Where the method comes from

This isn't textbook PCA — SVD wasn't covered in class — so the methodology is taken directly from Schuessler et al. (2020), *The interplay between randomness and structure during learning in RNNs* (NeurIPS 2020). Their **Equation (1)** decomposes the trained recurrent matrix as

$$W = W_0 + \Delta W$$

where $W_0$ is the random/baseline initialization and $\Delta W$ is the change induced by training. Their **Equation (4)** then defines the truncated SVD of the learned perturbation:

$$\Delta W^{(R)} = \sum_{r=1}^{R} s_r \mathbf{u}_r \mathbf{v}_r^T$$

Their **Figure 1(g–i)** shows the singular-value spectrum of $\Delta W$ for three neuroscience tasks. The reason for studying $\Delta W$ rather than $W$ directly: $W_0$ is a full-rank random matrix that adds a long tail to the SV spectrum of $W$, washing out whatever low-rank structure learning produces. Subtracting $W_0$ isolates the learning signal.

We apply this exact framework, parameterized by distractor intensity instead of task type. For each finetuned network, $W_0$ is the matched-seed λ=0 baseline (which is what each finetune was forked from), and $\Delta W$ is the subsequent change induced by training under distractors.

### The intuition

PCA looks at the *activity* (what the network does during a trial). SVD on $\Delta W$ looks at *what training under distraction changed about the wiring*, isolated from the random structure that was already there at initialization.

$\Delta W$ is a 128×128 matrix. SVD decomposes it into a list of "modes" — each mode is a pattern of input/output coupling that learning amplified, and the singular value tells you how strongly. A few large singular values plus many small ones means learning produced a low-rank perturbation: distractor exposure essentially edited a small number of dominant patterns. Many comparable singular values would mean learning made diffuse, unstructured changes.

### What we compute

**Reads:** `finetuned_lam{L}_seed{s}/target95.pt` (W_trained) and `lam0_seed{s}/epoch_100.pt` (W₀, the matched-seed baseline used as the finetune's initialization). Both store `W_rec` under the key `W_hh`.
**Writes:** `svd_spectrum_vs_lambda.png`.

1. Load both checkpoints, extract `W_hh` from each.
2. Compute $\Delta W = W_{\text{trained}} - W_0$ — a 128×128 matrix.
3. `np.linalg.svd(ΔW, compute_uv=False)` → 128 singular values sorted descending.
4. Repeat for all 3 seeds × 5 λ values, average across seeds, plot one curve per λ on log–log axes.

λ=0 is excluded from this analysis because there's no W_trained to compare against W₀ (they're the same checkpoint).

### What the plot shows

`svd_spectrum_vs_lambda.png`: the singular-value curves of $\Delta W$ for λ ∈ {2, 4, 6, 8, 10}, averaged across seeds.

Two patterns are visible. **First**, the curves separate cleanly by λ at the top — top singular value rises from ~2.0 at λ=2 to ~3.0 at λ=10. The Frobenius norm $\|\Delta W\|_F$ shows the same trend (~4.3 at λ=2, ~6–7 at λ=10). **Second**, the *shape* of the spectrum (the rate of decay across SV index) is roughly preserved across λ.

### What this means

Distractor exposure during finetuning produces a *larger-magnitude* learning-induced perturbation as λ increases — the network has to make bigger weight adjustments to reach 95% accuracy under stronger distraction. But the *rank distribution* of those adjustments is similar across λ: every distractor level produces a perturbation with the same characteristic shape, just scaled.

In Schuessler's terms: learning under stronger distraction recruits roughly the same number of dominant connectivity modes, but pushes them harder. Combined with the PR result (Step 1) and the ablation result (Step 3), the picture is that distractor exposure scales up the learned wiring change without fundamentally restructuring it.

---

## Step 3 — `testAblation.py`: Random unit ablation

### Where the method comes from

Ablation also wasn't covered in class. We follow the protocol used by Yang et al. (2019) — the same paper our continuous-time RNN architecture is based on — who silenced subsets of hidden units in their trained networks to test how task representations are distributed. They asked which units were critical for which tasks; we ask the simpler version of the same question: *if we silence a random subset of units, how does decision accuracy degrade, and does that degradation depend on the distractor intensity the network was trained on?*

### The intuition

If a network distributes its memory across many redundant pathways, killing some neurons shouldn't matter much — the remaining ones can pick up the slack. If it relies on a few critical neurons, killing them should be catastrophic.

So ablation directly tests *robustness*. We knock out a fraction of the 128 hidden units (set their activations to zero throughout the trial) and re-measure decision accuracy. We do this multiple times with different random subsets to average out luck.

### What we compute

**Reads:** `target95.pt` / `epoch_100.pt` (full state dict, used to rebuild the trained `VanillaRNN`) and `tasks/poisson_dms.generate_trials` (regenerates the same 50 validation trials deterministically from `seed * 1000 + 99`).
**Writes:** `ablation_vs_lambda.png`, `ablation_results.csv`.

For each network and each ablation fraction k ∈ {0%, 10%, 20%, 30%, 50%, 70%}:

1. Rebuild the model from the checkpoint and put it in eval mode.
2. Regenerate the same 50 validation trials (input X, label Y).
3. Pick a random subset of k×128 units to silence — build a binary mask of shape `(128,)` where chosen units = 0, others = 1.
4. Run the network forward with that mask. The mechanic is: after every recurrent update, multiply the hidden state by the mask. In the loop:
   ```python
   h = (1 - alpha) * h + alpha * tanh(W_in @ x[t] + W_rec @ h)
   h = h * mask    # silenced units forced to 0 every timestep
   ```
   So the ablated units can never contribute to subsequent dynamics.
5. Measure decision accuracy on the timesteps where label ≥ 1 (the test+decision period).
6. Repeat 5 times with different random subsets and average.

Final aggregation: for each λ we have 3 seeds × 5 random ablation realizations × 6 fractions = a grid of accuracies. Plot mean ± SEM per (λ, fraction).

### What the plot shows

`ablation_vs_lambda.png` is striking. With no ablation, all networks are ~95–100%. Once we ablate **even 10% of units**:

- λ=0 baseline holds at ~75% accuracy.
- λ=2 stays around ~74%.
- λ=4 drops to ~57%.
- λ=6, 8, 10 drop to ~38–45%.

By 30% ablation, λ=0 is still hovering near chance (~47%), while every distractor-trained network is well below.

### What this means

**Distractor-trained networks are dramatically more fragile to losing units.** This is probably the single most striking result in the suite.

Connect it back to PR: higher-λ networks use *more* dimensions (PR↑), but each dimension is apparently *more critical* (each unit harder to lose). It's the opposite of "robust redundancy" — it's more like a tightly coupled machine where every part is load-bearing. The λ=0 baseline, despite being lower-dimensional, is *more* robust because its few dimensions are spread across many redundant neurons.

So the original hypothesis ("more structure = more robust") is wrong in a specific, informative way. Distractor exposure produces *structure* in some senses (slightly lower-rank weights, more constrained activity patterns), but that structure is **brittle**, not resilient.

---

## Putting the whole story together

Three findings, one coherent narrative:

1. **Hidden-state dimensionality (PCA)** rises with λ. Memory representation gets *more* spread out across PCs as distractor intensity increases.
2. **Learning-induced weight perturbation (SVD on ΔW)** grows in magnitude with λ but keeps the same rank shape. Networks trained under stronger distraction make *bigger* weight changes from the baseline initialization, recruiting the same set of dominant connectivity modes more strongly rather than activating new ones.
3. **Robustness to ablation** drops sharply with λ. Networks trained under more distraction lose accuracy *much* faster when units are removed.

Reframed: distractor exposure during training produces networks that *converge* to the same task accuracy, but they get there by pushing harder along a similar set of connectivity modes (Step 2), spreading activity across more dimensions during the delay (Step 1), and ending up *specialized and brittle* rather than *redundant and robust* (Step 3). The original hypothesis assumed more distraction would lead to more compact, lower-dimensional, more robust representations. Empirically: the magnitude of learning *scales up*, dynamics *spread out*, and robustness *drops*.

All three results are reportable. The PR result and the ablation result are the clearest contributions; the SVD result connects them by showing that the changes in dynamics and robustness happen alongside larger learning-induced weight changes, not alongside a structural reorganization of the wiring.

## Reproduction

All four scripts are deterministic — the same checkpoints + the same code produce the same numbers every time. To regenerate everything:

```bash
git pull              # make sure you're on newlamdas with the latest commits
pip install numpy matplotlib scikit-learn torch
python analysis/testPR.py
python analysis/testSpectrum.py
python analysis/testSVD.py
python analysis/testAblation.py
```

Each produces its plots and CSVs in `analysis/`. The slowest one is `testAblation.py` (~2 minutes on CPU because it actually runs the model forward); the others are seconds.

## References for methods not covered in class

- **SVD on `W_rec` (Step 2).** Schuessler, F., Dubreuil, A., Mastrogiuseppe, F., Ostojic, S., & Barak, O. (2020). *The interplay between randomness and structure during learning in RNNs.* Advances in Neural Information Processing Systems, 33.
- **Random unit ablation (Step 3).** Yang, G. R., Joglekar, M. R., Song, H. F., Newsome, W. T., & Wang, X.-J. (2019). *Task representations in neural networks trained to perform many cognitive tasks.* Nature Neuroscience, 22(2), 297–306.

PCA itself (Step 1 / 1b) follows the standard six-step procedure covered in class, applied to the delay-period hidden-state covariance matrix; Participation Ratio is the specific summary statistic we use to turn PCA's eigenvalue output into a single scalar.
