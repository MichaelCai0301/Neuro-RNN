# Neuro-RNN: Teaching a Computer to Remember (Despite Distractions)

## What is this project?

Imagine you're shown a color, then you close your eyes for a few seconds while someone waves something distracting in front of you, and then you open your eyes and have to say whether a new color matches the original one. That's essentially what we're training a small artificial brain to do.

This project builds a **recurrent neural network (RNN)** — a type of artificial neural network that has a "memory" — and trains it on a classic neuroscience task called **Delayed Match-to-Sample with Distractors**. The task comes from a library called NeuroGym, which provides simulated neuroscience experiments.

## The Task, Step by Step

Each "trial" the network sees plays out like this:

1. **Fixation** — The network sees a "hold still" signal. Like a participant in a psychology experiment staring at a dot on a screen before the experiment begins. The network should do nothing (output 0).

2. **Sample** — A stimulus appears (think of it as a pattern or image). The network needs to pay attention and remember this. It still outputs 0 because no decision is needed yet.

3. **Delay** — The stimulus disappears. Now the network has to hold that memory in its head, with nothing to look at. This is the hard part — like remembering a phone number after someone tells you it.

4. **Distractor** — While the network is trying to remember, a random irrelevant stimulus appears to throw it off. Like someone shouting a different number at you while you're trying to remember the first one.

5. **Test** — A new stimulus appears. The network has to answer: "Is this the same as what I saw before?" If yes, output 1 (match). If no, output 0 (no match).
e
### What the data actually looks like

- **Inputs:** At each moment in time, the network receives 33 numbers. The first number is the "fixation signal" (1 = pay attention, 0 = make your decision). The other 32 numbers describe the stimulus using something called population coding — basically, different "sensor neurons" respond more or less strongly depending on the stimulus orientation.

- **Labels:** At each moment in time, there's a correct answer. For most of the trial, the correct answer is 0 (do nothing). Only during the test period does the network need to actually respond with 1 (match).

## The Model: A Tiny Artificial Brain

### Why an RNN?

Regular neural networks process each input independently — they have no memory. But our task *requires* memory: the network has to remember the sample stimulus across the delay period. An RNN solves this by having connections that loop back on themselves. At each moment, the network's internal state depends not just on the current input, but also on what it was thinking at the previous moment. This is similar to how neurons in the brain are connected to each other in loops.

### Why a "vanilla" RNN instead of fancier ones?

There are more powerful variants of RNNs called LSTMs and GRUs that are better at remembering things. But we chose a plain (vanilla) RNN on purpose:

- It has a single matrix of connection weights between its internal neurons (`W_hh`). This is directly analogous to synaptic connection strengths between real neurons.
- This simplicity lets us open up the network and study *how* it remembers — which neurons are connected to which, how activity patterns evolve over time, etc.
- Fancier models have extra gating machinery that makes this kind of analysis much harder.

### Architecture

Think of it as three layers:

```
Input (33 sensors) --> 64 "brain" neurons (connected to each other) --> Decision (match or no match)
```

- **Input layer:** 33 numbers come in (fixation + 32 stimulus channels)
- **Hidden layer:** 64 artificial neurons that talk to each other. Each neuron receives input from the stimulus AND from all the other neurons (including itself) from the previous timestep. This is where the "memory" lives.
- **Output layer:** Reads the hidden layer's activity and produces a decision — match (1) or no match (0).

The total number of learnable parameters (connection strengths the network adjusts during training) is roughly 6,400.

## How We Trained It

### The basic idea

Training a neural network is like teaching by repetition and feedback:
1. Show the network a trial (a sequence of inputs over time)
2. Let it make predictions at each timestep
3. Compare its predictions to the correct answers
4. Calculate how wrong it was (this is called the **loss**)
5. Slightly adjust all the connection strengths to make it a little less wrong next time
6. Repeat thousands of times

### Specific details

- **Training data:** 2,000 trials (separate random stimuli each time)
- **Validation data:** 400 trials the network never trains on — we use these to check that it's actually learning the task, not just memorizing the training examples
- **Optimizer:** Adam — a standard algorithm that adjusts the learning rate automatically for each connection weight. We used a learning rate of 0.0005 (small steps so training is stable).
- **Number of passes (epochs):** 100 — the network sees all 2,000 training trials 100 times
- **Batch size:** 64 — instead of updating weights after every single trial, we average the feedback over 64 trials at a time. This makes training smoother and faster.

### Dealing with class imbalance

Here's a tricky problem: during each trial, the network outputs a prediction at *every* timestep. But the actual decision only happens during a few timesteps at the end (the test period). So about 90% of timesteps have the "correct" answer of 0 (do nothing), and only about 10% require a real decision.

Without any correction, the network could get 90% accuracy by just *always* predicting 0 — never actually learning the task. To fix this, we give extra weight to the decision timesteps when calculating the loss. This tells the network: "Getting the decision right matters much more than getting the fixation period right."

### Gradient clipping

Vanilla RNNs have a known problem called **exploding gradients**: when the network processes long sequences, the error signals that flow backward through time can grow exponentially large, causing the weights to change wildly and training to blow up. We prevent this by capping (clipping) the gradients at a maximum size of 1.0.

### Checkpoints

Every 10 epochs, we save a snapshot of the network's weights. This lets us go back later and study how the network's "brain" changed over the course of learning — did certain connections get stronger? Did the network develop specialized groups of neurons?

## Results

| What we measured | Beginning (Epoch 1) | End (Epoch 100) |
|---|---|---|
| Training loss (how wrong the network is) | 0.67 | 0.28 |
| Overall accuracy (all timesteps) | 64.6% | 81.1% |
| Decision accuracy (just the test period) | 48.1% | **100.0%** |
| Validation loss | 0.62 | 0.28 |
| Validation decision accuracy | 43.7% | **100.0%** |

### What do these numbers mean?

- **Decision accuracy went from ~48% to 100%.** At the start, the network was basically guessing randomly (50/50 match vs. no match). By epoch 40 or so, it learned to get every single decision correct — on both the training data and the held-out validation data.

- **Overall accuracy is "only" 81%.** This might seem low, but it makes sense. The overall accuracy includes all the fixation timesteps, where the loss weighting tells the network not to worry too much. The network cares most about the timesteps that matter (the decisions), and it nails those.

- **Validation tracks training closely.** When a network does well on training data but poorly on new data, that's called **overfitting** (like memorizing answers to a practice test without understanding the material). Our validation performance matches training performance, which means the network genuinely learned the task rather than memorizing specific trials.

## What We Saved for Later Analysis

After training, we extracted the network's internal activity (the hidden states) on 50 validation trials. These hidden states show what the 64 neurons were doing at every moment during each trial. With these, we can:

- Use **PCA** (a technique for simplifying high-dimensional data) to visualize how the network's "thoughts" move through a low-dimensional space
- Compare how the network's internal activity differs between trials with and without distractors
- Check whether the memory representation stays stable during the delay period or drifts over time

## File Structure

- `rnn_model.py` — All the code: data generation, model definition, training loop, and hidden state extraction
- `data.ipynb` — A notebook we used to explore what the NeuroGym data looks like before building the model
- `checkpoints/distractor_1.0/` — Saved model snapshots (every 10 epochs), extracted hidden states, and a record of how training loss and accuracy changed over time

## Next Steps: Weight Matrix Extraction Across Distractor Conditions

The core idea is to train a separate network for each distractor strength (e.g., 0.0, 0.5, 1.0, 1.5, 2.0) and extract the 64×64 recurrent weight matrix `W_hh` from each. Since each network is trained from a different random initialization, comparing these weight matrices requires careful methodology.

### Per-Network Analysis

- **Eigenvalue/SVD decomposition of W_hh:** Decompose each network's recurrent weight matrix and compare eigenvalue spectra across conditions. Key things to look at: spectral radius (largest |eigenvalue|), fraction of eigenvalues with magnitude > 1, and the distribution of eigenvalues in the complex plane. Larger spectral radii indicate stronger recurrent amplification, which may relate to distractor resistance. *(Sussillo & Barak, Neural Computation 2013)*
- **Effective rank:** Compare how many significant singular values W_hh has across conditions. Networks handling stronger distractors may develop higher-rank or more structured connectivity.
- **Non-normal dynamics:** Compute the Schur decomposition of W_hh. Non-normality (large off-diagonal components) drives transient amplification, which may be relevant to how the network resists or succumbs to distractors. *(Goldman, Neuron 2009)*
- **Fixed-point analysis:** Use numerical optimization to find fixed points of the trained dynamics (where dx/dt ≈ 0), then linearize the Jacobian at each. Compare the number, stability, and geometry of fixed points across distractor conditions — this reveals whether stronger distractors change the attractor landscape. The `FixedPointFinder` library (Golub & Sussillo) provides a ready-made implementation. *(Sussillo & Barak, Neural Computation 2013)*

### Across-Network Comparison

Direct element-wise comparison of W_hh between separately trained networks is meaningless because neurons in different networks don't correspond to each other (the permutation invariance problem). Instead, use representation-level methods:

- **Centered Kernel Alignment (CKA):** Compares hidden-state representations rather than raw weights, making it invariant to neuron permutation and orthogonal transformations. Run all networks on a shared probe stimulus set and compute CKA between their hidden activations. *(Kornblith et al., ICML 2019)*
- **Representational Similarity Analysis (RSA):** Build representational dissimilarity matrices (RDMs) from each network's hidden-state responses to a shared stimulus set, then correlate RDMs across networks. *(Kriegeskorte et al., 2008)*
- **Dynamical Similarity Analysis (DSA):** Directly compares the dynamical systems learned by different networks using delay embeddings, invariant to neuron permutation. This is particularly well-suited for RNNs since it compares dynamics, not just static representations. *(Ostrow et al., NeurIPS 2023)*
- **Procrustes alignment:** Find the optimal orthogonal transformation aligning hidden states of two networks on the same inputs, then compare the aligned weight matrices. *(Williams et al., NeurIPS 2021)*

### Suggested Pipeline

1. Train one network per distractor condition (varying `distractor_strength`), saving W_hh at each checkpoint.
2. Per network: compute eigenspectrum of W_hh, run fixed-point analysis, measure effective rank.
3. Across networks: apply CKA or DSA on hidden activations to a shared probe stimulus set; compare eigenvalue spectra; compare fixed-point topology.
4. Look for systematic trends: does increasing distractor strength lead to predictable changes in spectral radius, attractor structure, or representational geometry?
