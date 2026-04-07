"""
RNN for Delayed Match-to-Sample with Distractors (Neuro 120 Project)

This script trains a simple recurrent neural network (RNN) to do a memory
task that neuroscientists commonly use to study working memory in animals
and humans.

The task works like this:
  1. The network sees a "sample" stimulus (like being shown a color)
  2. The stimulus disappears for a delay period (remember it!)
  3. A distractor stimulus appears to try to mess up the memory
  4. A "test" stimulus appears, and the network has to say whether
     it matches the original sample or not

Think of it like a memory card game, but with someone waving
distracting cards in your face while you're trying to remember.

We use a simple RNN (instead of fancier models like LSTM) because
its internal wiring is easy to study — each connection weight between
neurons is like a synaptic connection in a real brain circuit.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================================
# STEP 1: GENERATE THE DATA
# ============================================================================
# We use NeuroGym, a library that simulates neuroscience experiments.
# It generates input sequences (what the network "sees") and correct
# answers (what the network should output) for each trial.
# ============================================================================

def generate_trials(num_trials=1000, distractor_strength=1.0, seed=42):
    """
    Create a bunch of trials for the memory task.

    Each trial is a sequence of inputs over time (like a movie), along
    with the correct answer at each moment.

    Args:
        num_trials: How many trials to generate (more = better training,
                    but slower). Think of each trial as one "round" of
                    the memory game.
        distractor_strength: How strong the distracting stimulus is.
                             1.0 = normal, higher = harder to ignore.
        seed: A number that makes the randomness reproducible.
              Using the same seed always gives the same trials.

    Returns:
        X: The inputs, shaped (num_trials, timesteps, 33).
           33 = 1 fixation signal + 32 stimulus channels.
           The 32 channels use "population coding" — different
           channels respond more or less depending on the stimulus,
           similar to how real neurons respond to preferred orientations.
        Y: The correct answers, shaped (num_trials, timesteps).
           0 = "do nothing" (during fixation, sample, delay, distractor)
           1 = "match!" (only during the test period, when applicable)
    """
    from neurogym.envs.delaymatchsample import DelayMatchSampleDistractor1D

    # Set the random seed so we get the same data every time we run this.
    # This is important for reproducibility — anyone can re-run our code
    # and get the exact same results.
    rng = np.random.RandomState(seed)

    # Create the simulated experiment environment.
    # dt=20 means each timestep represents 20 milliseconds of simulated time.
    env = DelayMatchSampleDistractor1D(dt=20)

    # We'll collect all trials into these lists, then stack them into arrays.
    all_inputs = []
    all_labels = []

    for trial_number in range(num_trials):
        # Start a new trial. This randomly picks a new stimulus and sets up
        # the timing for fixation, sample, delay, distractor, and test periods.
        env.new_trial()

        # env.ob = what the network "sees" at each timestep. Shape: (T, 33)
        #   Column 0: fixation signal (1 = "wait", 0 = "respond now")
        #   Columns 1-32: stimulus channels (population-coded orientation)
        trial_input = env.ob.copy()

        # env.gt = the correct answer at each timestep. Shape: (T,)
        #   0 = no response needed (most of the trial)
        #   1 = "match" (only at the end, during the test period)
        trial_label = env.gt.copy()

        all_inputs.append(trial_input)
        all_labels.append(trial_label)

    # Stack all trials into a single big array.
    # np.stack turns a list of 2D arrays into one 3D array.
    X = np.stack(all_inputs)   # shape: (num_trials, T, 33)
    Y = np.stack(all_labels)   # shape: (num_trials, T)
    return X, Y


class TrialDataset(Dataset):
    """
    A helper class that wraps our data so PyTorch's DataLoader can
    automatically batch and shuffle it during training.

    PyTorch's DataLoader expects data to be in a "Dataset" object.
    This is just boilerplate — it converts our numpy arrays to PyTorch
    tensors and lets the DataLoader grab individual trials by index.
    """
    def __init__(self, X, Y):
        # torch.tensor() converts numpy arrays into PyTorch's format.
        # float32 for inputs (decimal numbers), long (integers) for labels.
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.long)

    def __len__(self):
        # How many trials are in this dataset?
        return self.X.shape[0]

    def __getitem__(self, idx):
        # Grab one trial by its index number.
        return self.X[idx], self.Y[idx]


# ============================================================================
# STEP 2: DEFINE THE NEURAL NETWORK
# ============================================================================
# Our network is a "vanilla" RNN — the simplest type of recurrent network.
#
# What makes it "recurrent"? At each timestep, the network's internal
# neurons receive two kinds of input:
#   1. The current stimulus (from the outside world)
#   2. The activity of all the other neurons from the PREVIOUS timestep
#
# This feedback loop is what gives the network memory — information from
# the past can persist in the network's activity patterns over time.
#
# The math at each timestep:
#   new_activity = tanh(W_input * current_stimulus + W_recurrent * previous_activity)
#
# W_input:     how strongly each input channel connects to each neuron
# W_recurrent: how strongly each neuron connects to every other neuron
#              (this is the key matrix — it's like a wiring diagram of the brain)
# tanh:        squashes the activity to be between -1 and 1 (keeps things stable)
# ============================================================================

class VanillaRNN(nn.Module):
    """
    A simple recurrent neural network for the memory task.

    Architecture:
        Input (33 channels) --> 64 recurrent neurons --> Decision (2 options)

    Why "vanilla" instead of LSTM or GRU?
        LSTMs and GRUs are better at remembering things over long delays,
        but they have complicated internal gating mechanisms that make it
        hard to interpret what's happening inside. Since our goal is to
        study HOW the network remembers (not just whether it can), we use
        the simplest possible recurrent network so we can directly look at
        the connection weights between neurons — just like a neuroscientist
        might study synaptic connections in a real brain.

    Why 64 neurons?
        This is a standard number in computational neuroscience papers.
        It's big enough for the network to develop interesting internal
        dynamics, but small enough that we can actually visualize and
        analyze what's going on inside.
    """

    def __init__(self, input_size=33, hidden_size=64, output_size=2):
        super().__init__()

        self.hidden_size = hidden_size

        # The recurrent layer: 64 neurons that are all connected to each other.
        #
        # Under the hood, this creates two weight matrices:
        #   W_ih (input-to-hidden):  how the 33 input channels connect to 64 neurons
        #   W_hh (hidden-to-hidden): how each of the 64 neurons connects to every
        #                            other neuron (including itself) — a 64x64 matrix
        #
        # W_hh is the most important matrix for our analysis. Each entry W_hh[i][j]
        # represents the connection strength from neuron j to neuron i, just like
        # a synaptic weight in a real neural circuit.
        self.rnn = nn.RNN(
            input_size=input_size,     # 33 input channels
            hidden_size=hidden_size,   # 64 internal neurons
            num_layers=1,              # just one layer of recurrent neurons
            nonlinearity='tanh',       # squash activity between -1 and 1
            batch_first=True,          # our data is shaped (batch, time, features)
        )

        # The readout layer: a simple linear function that looks at the 64 neurons'
        # activity and produces a decision.
        # Think of it as a "decoder" that reads the population activity and translates
        # it into an answer: class 0 (no match / do nothing) or class 1 (match).
        self.readout = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0=None):
        """
        Run the network on a batch of trials.

        This is the "forward pass" — we feed in the inputs and the network
        produces predictions. Later, we'll compare these predictions to the
        correct answers and adjust the weights (that's the "backward pass").

        Args:
            x:  Input data, shape (batch_size, timesteps, 33).
                A batch of trials, each one a sequence of 33-channel inputs.
            h0: Starting brain state, shape (1, batch_size, 64), or None.
                If None, all neurons start at zero activity — like a blank
                slate at the beginning of each trial.

        Returns:
            predictions:   shape (batch_size, timesteps, 2)
                           The network's guess at each timestep (raw scores
                           for each class, before converting to probabilities).
            hidden_states: shape (batch_size, timesteps, 64)
                           What all 64 neurons were doing at every moment.
                           We save this so we can study the network's internal
                           dynamics later (e.g., PCA visualization).
        """
        # Run the RNN through the entire sequence.
        # hidden_states: activity of all 64 neurons at every timestep
        # final_state: the very last hidden state (we don't need this)
        hidden_states, final_state = self.rnn(x, h0)

        # At each timestep, use the readout layer to convert the 64 neurons'
        # activity into a 2-class prediction (match vs. no match).
        predictions = self.readout(hidden_states)

        return predictions, hidden_states


# ============================================================================
# STEP 3: TRAIN THE NETWORK
# ============================================================================
# Training works by repeating this loop many times:
#   1. Show the network a batch of trials
#   2. Let it make predictions
#   3. Measure how wrong it was (the "loss")
#   4. Figure out which weights caused the errors (backpropagation)
#   5. Nudge each weight slightly to reduce the error
#
# Over many repetitions, the weights gradually adjust until the network
# learns to do the task correctly.
# ============================================================================

def train_model(
    model,
    train_loader,
    val_loader=None,
    num_epochs=50,
    learning_rate=1e-3,
    checkpoint_dir='checkpoints',
    checkpoint_every=10,
    device='cpu',
    class_weights=None,
):
    """
    Train the RNN on the memory task.

    Args:
        model:            The neural network to train.
        train_loader:     Training data, automatically batched and shuffled.
        val_loader:       Validation data (optional). Used to check that the
                          network is learning the task in general, not just
                          memorizing the specific training examples.
        num_epochs:       How many times to loop through all the training data.
                          One "epoch" = one complete pass through every trial.
        learning_rate:    How big of a step to take when adjusting weights.
                          Too big = training is unstable. Too small = too slow.
                          0.0005 is a reasonable middle ground.
        checkpoint_dir:   Folder to save snapshots of the network during training.
        checkpoint_every: Save a snapshot every N epochs (so we can look at how
                          the network's weights evolved over time).
        device:           'cpu' or 'cuda' (GPU). Use 'cpu' unless you have a GPU.
        class_weights:    How much to care about each type of error (see below).

    Returns:
        history: A dictionary tracking loss and accuracy over training.
    """
    # Create the folder to save checkpoints in.
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Move the model to the right device (CPU or GPU).
    model = model.to(device)

    # --- Set up the loss function ---
    # The loss function measures how wrong the network's predictions are.
    # We use "cross-entropy loss," which is standard for classification tasks.
    #
    # IMPORTANT: We use class weights to handle a tricky problem.
    # During each trial, the correct answer is "do nothing" (class 0) for ~90%
    # of the timesteps. Only ~10% of timesteps require a real decision (class 1).
    # Without weighting, the network can cheat by ALWAYS predicting "do nothing"
    # and still get 90% accuracy — without learning anything useful.
    #
    # The class weights tell the network: "Getting the decision timesteps right
    # is much more important than getting the fixation timesteps right."
    # We weight each class inversely by how common it is.
    loss_function = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )

    # --- Set up the optimizer ---
    # The optimizer decides HOW to adjust the weights based on the gradients.
    # Adam is a popular choice because it adapts the step size for each weight
    # individually — weights that need bigger adjustments get bigger steps.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # We'll track these metrics over time so we can plot learning curves.
    history = {
        'train_loss': [],       # how wrong the network is (lower = better)
        'train_acc': [],        # % of ALL timesteps predicted correctly
        'train_dec_acc': [],    # % of DECISION timesteps predicted correctly
        'val_loss': [],         # same metrics on data the network hasn't seen
        'val_acc': [],
        'val_dec_acc': [],
    }

    for epoch in range(1, num_epochs + 1):

        # === TRAINING PHASE ===
        # Tell the model we're in training mode (this affects some internal
        # behaviors like dropout, though we're not using dropout here).
        model.train()

        # Accumulators for this epoch's metrics.
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        decision_correct = 0    # how many decision timesteps we got right
        decision_total = 0      # how many decision timesteps there were

        # Loop through the training data in batches of 64 trials.
        for inputs_batch, labels_batch in train_loader:
            # Move data to the right device (CPU or GPU).
            inputs_batch = inputs_batch.to(device)
            labels_batch = labels_batch.to(device)

            # STEP 1: Forward pass — feed the inputs through the network.
            # The network produces raw scores (logits) for each class
            # at each timestep. We ignore the hidden states during training
            # (the underscore _ means "throw this away").
            predictions, _ = model(inputs_batch)

            # STEP 2: Calculate the loss (how wrong were the predictions?).
            # We need to reshape the data because the loss function expects
            # a flat list of predictions, not a 3D tensor.
            #   predictions: (batch, time, 2) --> flatten to (batch*time, 2)
            #   labels:      (batch, time)    --> flatten to (batch*time,)
            loss = loss_function(
                predictions.reshape(-1, predictions.size(-1)),
                labels_batch.reshape(-1),
            )

            # STEP 3: Backward pass — compute gradients.
            # This figures out how much each weight contributed to the error.
            # First, zero out any leftover gradients from the previous batch.
            optimizer.zero_grad()
            # Then compute new gradients by backpropagating the error.
            loss.backward()

            # STEP 3.5: Gradient clipping.
            # Vanilla RNNs have a known problem: when processing long sequences,
            # the gradients can grow exponentially large as they flow backward
            # through time (called "exploding gradients"). This causes the weights
            # to change wildly and training to become unstable.
            # Gradient clipping caps the gradient size at 1.0 to prevent this.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # STEP 4: Update the weights.
            # The optimizer nudges each weight in the direction that reduces
            # the loss, by an amount proportional to the learning rate.
            optimizer.step()

            # --- Track how the network is doing ---

            # Overall loss for this batch.
            running_loss += loss.item() * inputs_batch.size(0)

            # Which class did the network predict at each timestep?
            # .argmax(dim=-1) picks the class with the highest score.
            predicted_classes = predictions.argmax(dim=-1)

            # How many timesteps did it get right overall?
            correct_predictions += (predicted_classes == labels_batch).sum().item()
            total_predictions += labels_batch.numel()

            # How many DECISION timesteps did it get right?
            # This is the metric we really care about — can the network
            # actually tell match from non-match when it matters?
            is_decision_timestep = (labels_batch == 1)
            if is_decision_timestep.any():
                decision_correct += (predicted_classes[is_decision_timestep] == 1).sum().item()
                decision_total += is_decision_timestep.sum().item()

        # Compute averages for this epoch.
        avg_train_loss = running_loss / len(train_loader.dataset)
        overall_accuracy = correct_predictions / total_predictions
        decision_accuracy = (decision_correct / decision_total
                             if decision_total > 0 else 0.0)

        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(overall_accuracy)
        history['train_dec_acc'].append(decision_accuracy)

        # === VALIDATION PHASE ===
        # Test the network on data it has never trained on, to make sure
        # it's actually learning the task and not just memorizing.
        val_loss, val_acc, val_dec_acc = None, None, None
        if val_loader is not None:
            val_loss, val_acc, val_dec_acc = evaluate(
                model, val_loader, loss_function, device
            )
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_dec_acc'].append(val_dec_acc)

        # Print a progress update.
        msg = f"Epoch {epoch:3d}/{num_epochs} | "
        msg += f"Loss: {avg_train_loss:.4f}  DecAcc: {decision_accuracy:.4f}"
        if val_loss is not None:
            msg += f" | ValLoss: {val_loss:.4f}  ValDecAcc: {val_dec_acc:.4f}"
        print(msg)

        # === SAVE A CHECKPOINT ===
        # Every few epochs, save a snapshot of the network's current state.
        # This lets us go back later and study how the weights changed over
        # the course of training — like taking time-lapse photos of the
        # network's "brain" as it learns.
        if epoch % checkpoint_every == 0 or epoch == num_epochs:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'train_acc': overall_accuracy,
                'val_loss': val_loss,
                'val_acc': val_acc,
                # Save the recurrent weight matrix separately for easy access.
                # This is the 64x64 "wiring diagram" — the key thing we want
                # to analyze later.
                'W_hh': model.rnn.weight_hh_l0.detach().cpu().numpy(),
                'W_ih': model.rnn.weight_ih_l0.detach().cpu().numpy(),
            }
            path = os.path.join(checkpoint_dir, f'epoch_{epoch:03d}.pt')
            torch.save(checkpoint, path)
            print(f"  -> Checkpoint saved: {path}")

    return history


def evaluate(model, loader, loss_function, device='cpu'):
    """
    Test the network on a dataset WITHOUT updating any weights.
    Used to check performance on validation data.

    Returns: (average loss, overall accuracy, decision accuracy)
    """
    # Tell the model we're in evaluation mode (not training).
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0
    decision_correct = 0
    decision_count = 0

    # torch.no_grad() tells PyTorch not to track gradients during evaluation.
    # This saves memory and speeds things up — we're just measuring performance,
    # not training.
    with torch.no_grad():
        for inputs_batch, labels_batch in loader:
            inputs_batch = inputs_batch.to(device)
            labels_batch = labels_batch.to(device)

            predictions, _ = model(inputs_batch)
            loss = loss_function(
                predictions.reshape(-1, predictions.size(-1)),
                labels_batch.reshape(-1),
            )

            total_loss += loss.item() * inputs_batch.size(0)
            predicted_classes = predictions.argmax(dim=-1)
            total_correct += (predicted_classes == labels_batch).sum().item()
            total_count += labels_batch.numel()

            is_decision_timestep = (labels_batch == 1)
            if is_decision_timestep.any():
                decision_correct += (predicted_classes[is_decision_timestep] == 1).sum().item()
                decision_count += is_decision_timestep.sum().item()

    avg_loss = total_loss / len(loader.dataset)
    accuracy = total_correct / total_count
    dec_accuracy = decision_correct / decision_count if decision_count > 0 else 0.0
    return avg_loss, accuracy, dec_accuracy


# ============================================================================
# STEP 4: EXTRACT HIDDEN STATES FOR ANALYSIS
# ============================================================================
# After training, we want to look inside the network and see what the
# 64 neurons were doing during each trial. This lets us study questions like:
#   - How does the network represent the "memory" of the sample stimulus?
#   - Does the distractor mess up the memory representation?
#   - Can we find low-dimensional patterns in the neural activity (using PCA)?
# ============================================================================

def extract_hidden_states(model, X, device='cpu'):
    """
    Run the trained network on some trials and record what all 64 neurons
    were doing at every timestep.

    Args:
        model: The trained RNN.
        X: Input data, shape (num_trials, timesteps, 33).

    Returns:
        hidden_states: shape (num_trials, timesteps, 64).
            The activity of every neuron at every moment during each trial.
    """
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    with torch.no_grad():
        _, hidden_states = model(X_tensor)

    return hidden_states.cpu().numpy()


# ============================================================================
# STEP 5: PUT IT ALL TOGETHER
# ============================================================================

def main():
    # --- Settings ---
    # You can adjust these to experiment with different configurations.

    NUM_TRAIN_TRIALS = 2000    # how many trials to train on
    NUM_VAL_TRIALS = 400       # how many trials to validate on (separate data)
    BATCH_SIZE = 64            # how many trials to process at once
    NUM_EPOCHS = 100           # how many times to loop through all training data
    LEARNING_RATE = 5e-4       # how aggressively to adjust weights (smaller = safer)
    HIDDEN_SIZE = 64           # how many neurons in the recurrent layer
    CHECKPOINT_EVERY = 10      # save a snapshot every N epochs
    DISTRACTOR_STRENGTH = 1.0  # how strong the distractor is (1.0 = default)

    # Use GPU if available, otherwise CPU.
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")

    # --- Generate the training and validation data ---
    print("Generating training trials...")
    X_train, Y_train = generate_trials(
        num_trials=NUM_TRAIN_TRIALS,
        distractor_strength=DISTRACTOR_STRENGTH,
        seed=42,    # fixed seed for reproducibility
    )
    print(f"  Input shape:  {X_train.shape}")    # (2000, T, 33)
    print(f"  Labels shape: {Y_train.shape}")    # (2000, T)
    print(f"  Unique labels: {np.unique(Y_train)}")

    print("Generating validation trials...")
    X_val, Y_val = generate_trials(
        num_trials=NUM_VAL_TRIALS,
        distractor_strength=DISTRACTOR_STRENGTH,
        seed=99,    # different seed so validation data is different from training
    )

    # --- Handle class imbalance ---
    # Count how many timesteps belong to each class.
    # Class 0 (fixation/do nothing) will vastly outnumber class 1 (match).
    num_classes = len(np.unique(Y_train))
    class_counts = np.bincount(Y_train.flatten(), minlength=num_classes)
    total_samples = class_counts.sum()

    # Compute weights: rare classes get higher weight so the network pays
    # more attention to getting them right.
    class_weights = torch.tensor(
        total_samples / (num_classes * class_counts),
        dtype=torch.float32,
    )
    print(f"  Class counts: {dict(enumerate(class_counts))}")
    print(f"  Class weights: {class_weights.tolist()}")

    # Wrap the data in DataLoaders for automatic batching and shuffling.
    train_dataset = TrialDataset(X_train, Y_train)
    val_dataset = TrialDataset(X_val, Y_val)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    # --- Create the model ---
    model = VanillaRNN(
        input_size=33,              # 33 input channels
        hidden_size=HIDDEN_SIZE,    # 64 recurrent neurons
        output_size=num_classes,    # 2 output classes (no match vs. match)
    )
    print(f"\nModel architecture:\n{model}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable parameters: {total_params:,}")

    # --- Train the model ---
    checkpoint_dir = os.path.join(
        'checkpoints', f'distractor_{DISTRACTOR_STRENGTH}'
    )
    history = train_model(
        model,
        train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=CHECKPOINT_EVERY,
        device=DEVICE,
        class_weights=class_weights,
    )

    # --- Save hidden states for later analysis ---
    # After training, we run the trained network on 50 validation trials
    # and save what all 64 neurons were doing at every timestep.
    # This is the data we'll use for PCA and other analyses.
    print("\nExtracting hidden states from trained model...")
    hidden_states = extract_hidden_states(model, X_val[:50], device=DEVICE)
    print(f"  Hidden states shape: {hidden_states.shape}")

    np.save(
        os.path.join(checkpoint_dir, 'hidden_states_val.npy'),
        hidden_states,
    )
    print(f"  Saved to {checkpoint_dir}/hidden_states_val.npy")

    # --- Save the training history ---
    # This records how loss and accuracy changed over training, so we can
    # plot learning curves later.
    np.savez(
        os.path.join(checkpoint_dir, 'training_history.npz'),
        train_loss=history['train_loss'],
        train_acc=history['train_acc'],
        train_dec_acc=history['train_dec_acc'],
        val_loss=history['val_loss'],
        val_acc=history['val_acc'],
        val_dec_acc=history['val_dec_acc'],
    )
    print("Done! Training history saved.")


if __name__ == '__main__':
    main()
