# AI Usage Log

This document records which parts of the codebase were developed with Claude (AI assistant) and what role it played in each.

---

## 1. Training Pipeline with Class Imbalance Handling
**File:** `rnn_model.py` — `train_model()` (lines 94–203)

The full training loop was structured and built out with Claude's help. Key AI-contributed elements:

- **Class-weighted cross-entropy loss:** The strategy of weighting classes inversely by frequency to prevent the network from "cheating" by always predicting "do nothing" (which covers ~90% of timesteps). Claude identified this as a critical failure mode and implemented the `class_weights` parameter with `nn.CrossEntropyLoss(weight=...)`.
- **Decision-specific accuracy tracking:** Standard accuracy would be misleading (90%+ just from predicting class 0). Claude added separate tracking of decision-timestep accuracy (`dec_correct` / `dec_total`), filtering only timesteps where `label == 1`, which is the metric that actually measures whether the model learned the task.
- **Gradient clipping:** Claude added `clip_grad_norm_(max_norm=1.0)` to handle exploding gradients, a known instability in vanilla RNNs trained on long sequences via backpropagation through time.
- **Checkpoint saving with weight matrix extraction:** The checkpointing logic saves not just the model state, but explicitly extracts and stores `W_hh` (the 64x64 recurrent weight matrix) and `W_ih` as numpy arrays for downstream neuroscience analysis — this was designed with Claude to make the analysis pipeline easier.

## 2. Model Architecture Design (`VanillaRNN` class)
**File:** `rnn_model.py` — `VanillaRNN` (lines 59–91)

While the choice of a vanilla RNN was a human design decision (motivated by interpretability for neuroscience), Claude helped with:

- **Structuring the `nn.Module` subclass** with proper `__init__`/`forward` pattern, setting up `nn.RNN` with the right parameters (`batch_first=True`, `tanh` nonlinearity, single layer).
- **The readout layer design:** Using `nn.Linear(hidden_size, output_size)` applied at every timestep (not just the final one), so the network produces a prediction at each moment — necessary for the trial-structure where the "respond now" signal can come at different times.
- **Returning hidden states alongside predictions** from `forward()`, which is non-standard but essential for the neuroscience analysis (PCA on neural trajectories, etc.).

## 3. Evaluation and Validation Pipeline
**File:** `rnn_model.py` — `evaluate()` (lines 206–238)

Claude wrote the `evaluate()` function as a separated, `torch.no_grad()` evaluation loop that mirrors the training metrics:

- Computes loss, overall accuracy, and decision-specific accuracy on held-out validation data.
- Uses the same decision-timestep filtering logic as training to give an apples-to-apples comparison.
- Integrated into the training loop to run after each epoch, enabling monitoring for overfitting.

## 4. Hidden State Extraction for Neuroscience Analysis
**File:** `rnn_model.py` — `extract_hidden_states()` (lines 243–249) and `main()` (lines 324–328)

Claude built the post-training analysis pipeline:

- `extract_hidden_states()` runs the trained model on validation trials and captures the full `(num_trials, timesteps, 64)` tensor of neural activity — every neuron's activation at every moment.
- The `main()` function saves these as `.npy` files alongside training history (`.npz`), creating a clean data pipeline from training to downstream analysis (PCA, attractor dynamics, etc.).
- This design choice — separating hidden state extraction from training and saving to disk — was Claude's suggestion to keep the analysis modular.

---

**What was NOT AI-generated:**
- The experimental design choice (delayed match-to-sample with distractors)
- The use of NeuroGym's `DelayMatchSampleDistractor1D` environment
- Hyperparameter values (trial counts, learning rate, hidden size, etc.)
- The neuroscience motivation and research questions
- The `data_test.ipynb` notebook for data exploration
