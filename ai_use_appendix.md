# Project Contribution Breakdown

## Michael Cai

**Task and distractor design.** Led the research on how distractors should be represented mathematically, drawing on Hallenbeck et al. (2021) and reviewer feedback to determine that distractors should be competing stimuli on the same representational substrate (cosine bumps on the ring encoding) rather than additive noise or fidelity degradation. Implemented the initial NeuroGym-based task environment and data generation pipeline, including the `DelayMatchSampleDistractor1D` integration and trial metadata extraction.

**Fidelity decoder and analysis.** Designed and implemented the fidelity decoder — a ridge regression from hidden states to [sin(theta), cos(theta)] — to measure how well the remembered angle can be read out at each timestep during the delay period. Tested the decoder on trained networks to verify that fidelity remains high during clean trials and dips transiently around distractor onsets. This analysis is the primary tool for connecting our RNN results to the Hallenbeck et al. findings.

**Write-up and literature.** Took the lead on drafting the project write-up, including the background, methods, and results sections. Conducted the deeper literature review connecting our computational approach to the biological findings on working memory representations in visual cortex.

## Kyle Zheng

**Model architecture and training.** Built the initial vanilla RNN and iteratively debugged it through several failed architectures (nn.RNN with 64 units, 128 units, 2-layer, loss-masked variants) before arriving at the working continuous-time RNN formulation with tau=500ms. Diagnosed the label encoding bug where NeuroGym's `env.gt` only encoded timing, not match/non-match, and the memorization problem with discrete angles. Tuned hyperparameters (learning rate, hidden size, tau, class weights) to achieve 97% generalization accuracy on the baseline task.

**Training sweep and infrastructure.** Ran the full training sweep across lambda in {0, 0.5, 1, 2, 4} x 3 seeds = 15 networks, confirming all networks train above 97% accuracy. Built the sweep script, checkpoint management, and the self-contained notebook visualization pipeline (`inspect_trial`) that works without external dependencies.

**Debugging and integration.** Resolved dependency conflicts between Python environments, dimension mismatches when switching between task variants, and output class propagation issues. Built the custom `PoissonDistractorDMS` task (jointly with Michael's distractor design) to eliminate the NeuroGym dependency and support continuous angle sampling.

## Shared

**Research question, hypothesis, and experimental design.** Both members contributed to formulating the research question, selecting the Poisson distractor protocol, choosing lambda values for the sweep, and prioritizing analyses (fidelity decoder and PCA over ablation) in response to reviewer feedback. Regular exchange of findings and coordination on model structure and parameters throughout.

---

# AI Use Appendix

All of our AI use has been provided by Claude Code (Anthropic's CLI tool). For debugging and diagnosing training failures, our initial RNN was trained on NeuroGym's DelayMatchSampleDistractor1D task, which was stuck at 66% decision accuracy (always guessing non-match for one match and two non-match tests). Claude identified the root cause: that NeuroGym's `env.gt` only encoded when to respond (0 = hold fixation, 1 = respond now), not the actual what (match vs. non match). The actual match/non-match information was stored in `env.trial['ground_truth']`. After fixing our labels, the network plateaued at 50% accuracy, and Claude diagnosed that this was due to a vanishing gradient problem: PyTorch's `nn.RNN` uses `h_new = tanh(W*h + W*x)`, which overwrites the hidden state each timestep. This makes it nearly impossible to maintain information across the 125-step delay period. This prompted us to switch to a new continuous-time (leaky integrate) RNN formulation: `h_new = (1-alpha)*h + alpha*tanh(...)` based on Yang et al. (2019).

When we changed task environments, model architectures, or object dimensions, Claude helped identify which tensor shapes, loss configurations, and mathematical implementations (rounding, datatypes, function use, etc.) needed to change. Claude also resolved a dependency conflict where NeuroGym was available in one Python environment but not another, and it was suggested we decouple the notebook from NeuroGym entirely by pre-generating data. Claude saved us the most time here.

Last, it was very useful for understanding mathematical concepts in paper for implementation. It helped break down the math in papers, like in Hallenbeck et al. (2021), where Claude clarified the distinction between their inverted encoding model fidelity metric and standard decoding accuracy. For Yang et al. (2019), Claude walked through how the continuous-time formulation differs from a discrete-time RNN and why time constant tau is necessary for tasks with long delays.

## SVD, ablation, and the eigenvalue spectrum (methods not covered in class)

PCA was the only one of our final analyses that we covered in class, so for SVD on the recurrent weight matrix, random unit ablation, and the cumulative-variance / eigenvalue spectrum view of PCA, we had to work out the methods on our own using the literature. Claude was useful here as a scaffolding partner. Once we had decided what each analysis should measure and why, Claude helped translate that into the per-network loop, the seed-aggregation logic, and the small bookkeeping pieces (loading `W_rec` from a saved checkpoint, picking sensible default ranges for ablation fractions, etc.) that we hadn't written in numpy before.

For ablation specifically, the literature was Yang et al. (2019), which was the same paper our continuous-time architecture is based on, which already silences subsets of hidden units to test how task representations are distributed. We noticed this ourselves while re-reading the paper and decided it was the right way to test robustness across our distractor levels. Claude helped us think through one implementation question: whether to apply the silencing mask only at the readout (post-hoc) or inside the recurrent loop at every timestep so ablated units can't influence subsequent dynamics. We picked the second version after talking through why the first wouldn't actually test what we cared about: once a unit is silenced it shouldn't be allowed to drive the rest of the network on the next step.

For SVD, the methodology comes directly from Schuessler et al. (2020). Their Equation (1) splits the trained recurrent matrix as $W = W_0 + \Delta W$, and their Equation (4) defines the truncated SVD of the learned perturbation $\Delta W = \sum_r s_r \mathbf{u}_r \mathbf{v}_r^T$, with their Figure 1(g–i) showing the singular-value spectrum of $\Delta W$ across three neuroscience tasks. We initially wrote a simpler version that ran SVD on the trained weight matrix $W$ directly, but Claude flagged that this misses the point of Schuessler's framework: the random initialization $W_0$ is full-rank and adds a long tail to the spectrum that washes out the learning signal. After fetching the paper to verify the methodology, we updated `testSVD.py` to subtract the matched-seed baseline checkpoint from each finetuned checkpoint — possible only because each `finetuned_lam{L}_seed{s}/target95.pt` was forked from `lam0_seed{s}/epoch_100.pt`, so $W_0$ for every finetuned network is exactly available on disk. Beyond the methodology fix, Claude's contribution was mechanical: noting that the training loop saves the recurrent weight matrix under the key `W_hh` (separate from the full state dict) and helping pick log-log axes for the singular-value plot.

For the eigenvalue spectrum, the work was just running the same PCA we'd already done for the participation ratio and presenting the eigenvalues as a curve instead of collapsing them into one number. Claude pointed out that this is the textbook "scree plot" / cumulative-variance-explained format that PCA references.
