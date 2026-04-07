# Proposal Revision — Neuro 120 Course Project

**Group members:** Kyle Zheng, Michael Cai

---

## 1. Revised Proposal

### What we changed and why

**Clarifying what distractors are (Reviewer 1's feedback)**

Reviewer 1 asked us to be more specific about what our distractors consist of, since different types of noise have different consequences for decoding. After reading more about how distractors are typically modeled in working memory RNN studies, we want to clarify:

In NeuroGym's `DelayMatchSampleDistractor1D`, the distractor is a **task-relevant stimulus** — a different orientation presented through the same 32 population-coded channels during the delay period. This means the distractor directly competes with the memory representation because it uses the same input format as the sample. This is the most interesting type of distractor for our question, because it tests whether the network can maintain a memory trace even when a competing signal arrives through the same channels.

For our main experiment, we plan to vary the **strength** of this default distractor by scaling it (e.g., 0.5x, 1.0x, 2.0x). This is already supported by the `distractor_strength` parameter in our code. If time permits, we may also try adding unstructured Gaussian noise as a comparison condition, but the scaled task-relevant distractor is our primary manipulation.

**Separating PCA from ablation (Reviewer 2's feedback)**

Reviewer 2 pointed out that our original Analysis 2 mixed together ablation and PCA in a confusing way. We agree — these are really two different analyses. We've separated them:

- **Analysis 1** (unchanged): Hidden state stability during the delay period — measuring whether the network's internal representation drifts or stays stable while remembering.
- **Analysis 2** (clarified): PCA on hidden state trajectories — visualizing how the network's activity moves through a low-dimensional space during different trial periods. This is separate from ablation and serves as our main tool for understanding the network's dynamics.
- **Analysis 3** (clarified): Unit ablation — silencing individual units to see which ones matter most for the task. This is now a standalone analysis with clearer expected outcomes.

**Acknowledging the scope of dynamics comparison (Reviewer 1)**

Reviewer 1 noted that comparing dynamics and weights across RNNs is its own subfield. We appreciate this heads-up. For now, we're sticking with PCA as our primary dynamics tool since it's well-established and we understand it. We'll explore other methods as we get more comfortable with the data, but we're not committing to anything we haven't tried yet.

---

## 2. Demonstration of Feasibility

### What we have working

Our code is on GitHub (github.com/MichaelCai0301/Neuro-RNN). We have a complete working pipeline:

- **Data generation** using NeuroGym's `DelayMatchSampleDistractor1D` environment (2,000 training trials, 400 validation trials)
- **Model:** A vanilla RNN with 33 inputs, 64 hidden units, and a 2-class readout (~6,400 parameters)
- **Training loop** with class weighting (to handle the imbalance between fixation and decision timesteps), gradient clipping, and checkpoint saving every 10 epochs
- **Hidden state extraction** for later analysis

### Training results (distractor strength = 1.0)

| Metric                       | Epoch 1  | Epoch 100  |
|------------------------------|----------|------------|
| Training loss                | 0.6701   | 0.2780     |
| Training decision accuracy   | 48.1%    | **100.0%** |
| Validation loss              | 0.6238   | 0.2811     |
| Validation decision accuracy | 43.7%    | **100.0%** |

The network starts at chance (~50%) on decision timesteps and reaches 100% accuracy by around epoch 40. Validation tracks training closely (no overfitting). This confirms the pipeline works and the model can learn the task. We can now move on to training under different distractor strengths.

---

## 3. Execution Plan

### Step 1: Train networks under different distractor strengths
**Who:** Kyle (runs training) + Michael (sets up distractor conditions)
**Timeline:** Week 1

Train separate networks at distractor strengths of 0.0 (no distractor), 0.5, 1.0 (already done), and 2.0, keeping all other settings the same.

**Expected result:** All networks should learn the task, but stronger distractors may slow convergence or require more epochs. Our hypothesis predicts that networks trained with stronger distractors will develop more structured internal connectivity to compensate. This is consistent with Ghazizadeh & Ching (2021), who showed that RNNs develop efficient low-dimensional dynamics for working memory tasks — we expect distractor pressure to amplify this effect.

**What success looks like:** All networks reach >90% decision accuracy with saved checkpoints across all conditions.

### Step 2: Visualize hidden state trajectories with PCA
**Who:** Michael (lead) + Kyle (support)
**Timeline:** Week 1–2

Apply PCA to the hidden states from each trained network. Plot the first 2–3 principal components over time, color-coded by trial period (fixation, sample, delay, distractor, test).

**Expected result:** We expect to see distinct trajectories for different trial periods. During the delay, activity should settle into a stable pattern representing the remembered stimulus. Networks trained with stronger distractors should show more clearly separated trajectories — reflecting a more robust memory representation that resists interference. This aligns with Murray et al. (2016), who found that stable working memory coding coexists with structured, low-dimensional population dynamics in prefrontal cortex.

**What success looks like:** Readable PCA plots that show visually distinguishable patterns across trial periods and across distractor conditions.

### Step 3: Measure hidden state stability during the delay
**Who:** Kyle (lead) + Michael (support)
**Timeline:** Week 2

For each trial, measure how much the hidden state drifts during the delay period (e.g., cosine similarity between the hidden state at the start of the delay and at later delay timesteps).

**Expected result:** Networks trained with stronger distractors should show more stable representations during the delay (higher cosine similarity over time), because training forced them to maintain memory despite interference. Networks trained without distractors may drift more since they were never pressured to be robust. This connects to our original hypothesis and to Ghazizadeh & Ching (2021)'s finding that RNNs develop slow, stable manifolds for memory maintenance.

**What success looks like:** A clear plot showing stability over delay time for each distractor condition, with a visible trend across conditions.

### Step 4: Unit ablation
**Who:** Kyle (lead) + Michael (support)
**Timeline:** Week 2–3

Silence one hidden unit at a time and measure how much decision accuracy drops for each network.

**Expected result:** If the network has developed specialized memory-supporting units, silencing those units will cause big accuracy drops while silencing others won't matter much. If memory is spread evenly across all units, every unit should matter about the same amount. We hypothesize that networks trained under stronger distraction will show more specialization — a few critical units rather than distributed memory — because the pressure to resist distractors may force the network to concentrate its memory function. This relates to Chaisangmongkon et al. (2017), who found functionally specialized subpopulations in RNNs trained on cognitive tasks.

**What success looks like:** A ranked bar chart of accuracy drop per unit for each condition, showing whether the distribution is skewed (specialized) or flat (distributed).

### Step 5: Weight matrix comparison (if time permits)
**Who:** Michael (lead) + Kyle (support)
**Timeline:** Week 3

Compare the recurrent weight matrices (W_hh) across distractor conditions. Start with basic visualizations (heatmaps) and look at whether the weight structure becomes more organized under stronger distractor training.

**Expected result:** Networks trained under stronger distraction should develop more structured W_hh matrices (less random-looking, potentially with identifiable clusters or patterns). This is exploratory — as Reviewer 1 noted, weight comparison is a deep subfield, so we'll start simple and go further if the initial results are interesting.

---

### Works cited (new to this revision)

- Chaisangmongkon, W., Swaminathan, S. K., Freedman, D. J., & Wang, X.-J. (2017). "Computing by robust transience: how the fronto-parietal network performs sequential, category-based decisions." *Neuron*, 93(6), 1504–1517.
