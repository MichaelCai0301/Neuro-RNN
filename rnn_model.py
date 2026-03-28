"""
This is kind of a proof of concept vanilla test RNN that trains on a working memory task where the network must
remember a stimulus, ignore a distractor (simple distractor as of now), and decide if a test stimulus
matches the original. 
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from neurogym.envs.delaymatchsample import DelayMatchSampleDistractor1D
# used from NeuroGym: pre-built neuroscience experiment that handles trial
# structure and encodes stimuli using cosine tuning curves across 32
# channels, which mimics how real neurons represent orientation

def generate_trials(num_trials=1000, distractor_strength=1.0):
    """Generate trials for the memory task using NeuroGym."""

    seed = 42 # 67 
    rng = np.random.RandomState(seed)
    env = DelayMatchSampleDistractor1D(dt=20) # dt determines how long each delay / test period is (the test is like fixation + sample + delay1 + test1 + delay2 + test2 + delay3 + test3)
    # also, the sample here is Unif(0, 2pi) and is the angle of the stimulus
    # fixation tells the network to stay still and not respond yet

    all_inputs = []
    all_labels = []

    for _ in range(num_trials):
        env.new_trial()
        # reminder of format:
        # env.ob: network input at each timestep, shape (T, 33)
        # column 0 = fixation signal, columns 1-32 = population-coded stimulus
        # env.gt: correct label at each timestep, shape (T,)
        # 0 = do nothing, 1 = match
        all_inputs.append(env.ob.copy())
        all_labels.append(env.gt.copy())

    X = np.stack(all_inputs) # (num_trials, T, 33)
    Y = np.stack(all_labels) # (num_trials, T)
    return X, Y


class TrialDataset(Dataset):
    """Wraps trial data so PyTorch can batch and shuffle it during training."""

    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# one forward pass through the RNN performs one working memory matching task
class VanillaRNN(nn.Module):
    """
    Input = 33 -> 64 recurrent neurons -> Decision (2)

    We use a vanilla RNN (not more advanced versions like LSTM) so the recurrent weight matrix W_hh
    directly represents neuron-to-neuron connection strengths, making it easy
    to analyze like a biological wiring diagram. 64 hidden units is standard
    in computational neuroscience it appears and is not super hard to study.
    """

    def __init__(self, input_size=33, hidden_size=64, output_size=2):
        super().__init__()
        self.hidden_size = hidden_size
        # Recurrent layer with two weight matrices:
        # W_ih: input-to-neuron connections (33 x 64)
        # W_hh: neuron-to-neuron connections (64 x 64), our main analysis target
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            nonlinearity='tanh',
            batch_first=True,
        )

        # readout layer reads population activity, outputs a decision
        self.readout = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0=None):
        """Run input through the network. Returns predictions and hidden states."""
        hidden_states, _ = self.rnn(x, h0)
        predictions = self.readout(hidden_states)
        return predictions, hidden_states


# Training
def train_model(
    model, train_loader, val_loader=None, num_epochs=50, learning_rate=1e-3,
    checkpoint_dir='checkpoints', checkpoint_every=10, device='cpu',
    class_weights=None,
):
    """Train the RNN and saves checkpoints"""

    os.makedirs(checkpoint_dir, exist_ok=True)
    model = model.to(device)

    # Cross-entropy loss with class weighting -> important so the network can't cheat by always predicting "do nothing" (90% of timesteps)
    loss_function = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )

    # we use default Adam optimizer for now
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        'train_loss': [], 'train_acc': [], 'train_dec_acc': [],
        'val_loss': [], 'val_acc': [], 'val_dec_acc': [],
    }

    for epoch in range(1, num_epochs + 1):

        # Train on batches
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        dec_correct = 0
        dec_total = 0

        for inputs_batch, labels_batch in train_loader:
            inputs_batch = inputs_batch.to(device)
            labels_batch = labels_batch.to(device)

            # forward pass
            predictions, _ = model(inputs_batch)

            loss = loss_function(
                predictions.reshape(-1, predictions.size(-1)),
                labels_batch.reshape(-1),
            )

            # backward pass
            optimizer.zero_grad()
            loss.backward()

            # Clip gradients to prevent exploding gradients (common in vanilla RNNs)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Track metrics
            running_loss += loss.item() * inputs_batch.size(0)
            predicted_classes = predictions.argmax(dim=-1)
            correct += (predicted_classes == labels_batch).sum().item()
            total += labels_batch.numel()

            #  decision accuracy 
            is_decision = (labels_batch == 1)
            if is_decision.any():
                dec_correct += (predicted_classes[is_decision] == 1).sum().item()
                dec_total += is_decision.sum().item()

        # Record epoch metrics
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct / total
        train_dec_acc = dec_correct / dec_total if dec_total > 0 else 0.0

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_dec_acc'].append(train_dec_acc)

        # Validate
        val_loss, val_acc, val_dec_acc = None, None, None
        if val_loader is not None:
            val_loss, val_acc, val_dec_acc = evaluate(
                model, val_loader, loss_function, device
            )
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_dec_acc'].append(val_dec_acc)

        # Print progress
        msg = f"Epoch {epoch:3d}/{num_epochs} | "
        msg += f"Loss: {train_loss:.4f}  DecAcc: {train_dec_acc:.4f}"
        if val_loss is not None:
            msg += f" | ValLoss: {val_loss:.4f}  ValDecAcc: {val_dec_acc:.4f}"
        print(msg)

        # Save checkpoint with recurrent weight matrix for analysis
        if epoch % checkpoint_every == 0 or epoch == num_epochs:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'W_hh': model.rnn.weight_hh_l0.detach().cpu().numpy(),
                'W_ih': model.rnn.weight_ih_l0.detach().cpu().numpy(),
            }
            path = os.path.join(checkpoint_dir, f'epoch_{epoch:03d}.pt')
            torch.save(checkpoint, path)
            print(f"  -> Checkpoint saved: {path}")

    return history


def evaluate(model, loader, loss_function, device='cpu'):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    dec_correct = 0
    dec_count = 0

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

            is_decision = (labels_batch == 1)
            if is_decision.any():
                dec_correct += (predicted_classes[is_decision] == 1).sum().item()
                dec_count += is_decision.sum().item()

    avg_loss = total_loss / len(loader.dataset)
    accuracy = total_correct / total_count
    dec_accuracy = dec_correct / dec_count if dec_count > 0 else 0.0
    return avg_loss, accuracy, dec_accuracy


# Hidden state extraction for analysis

def extract_hidden_states(model, X, device='cpu'):
    """Run the trained model and record all 64 neurons' activity over time."""
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        _, hidden_states = model(X_tensor)
    return hidden_states.cpu().numpy()


# Main

def main():
    # Settings for hyperparameters
    NUM_TRAIN_TRIALS = 2000
    NUM_VAL_TRIALS = 400
    BATCH_SIZE = 64
    NUM_EPOCHS = 100
    LEARNING_RATE = 5e-4
    HIDDEN_SIZE = 64
    CHECKPOINT_EVERY = 10
    DISTRACTOR_STRENGTH = 1.0

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")

    # Generate data using NeuroGym
    print("Generating training trials...")
    X_train, Y_train = generate_trials(
        num_trials=NUM_TRAIN_TRIALS,
        distractor_strength=DISTRACTOR_STRENGTH,
        seed=42,
    )
    print(f"Input shape:  {X_train.shape}")
    print(f"Labels shape: {Y_train.shape}")
    print(f"Unique labels: {np.unique(Y_train)}")

    print("Generating validation trials...")
    X_val, Y_val = generate_trials(
        num_trials=NUM_VAL_TRIALS,
        distractor_strength=DISTRACTOR_STRENGTH,
        seed=99,
    )

    # Compute class weights to handle imbalance
    # (most timesteps are "do nothing", only ~10% are actual decisions)
    num_classes = len(np.unique(Y_train))
    class_counts = np.bincount(Y_train.flatten(), minlength=num_classes)
    total_samples = class_counts.sum()
    class_weights = torch.tensor(
        total_samples / (num_classes * class_counts),
        dtype=torch.float32,
    )
    print(f"  Class counts: {dict(enumerate(class_counts))}")
    print(f"  Class weights: {class_weights.tolist()}")

    # Set up data loaders
    train_loader = DataLoader(
        TrialDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TrialDataset(X_val, Y_val), batch_size=BATCH_SIZE, shuffle=False
    )

    # Create and train the model
    model = VanillaRNN(
        input_size=33,
        hidden_size=HIDDEN_SIZE,
        output_size=num_classes,
    )
    print(f"\nModel architecture:\n{model}")
    print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters()):,}")

    checkpoint_dir = os.path.join('checkpoints', f'distractor_{DISTRACTOR_STRENGTH}')
    history = train_model(
        model, train_loader, val_loader=val_loader,
        num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE,
        checkpoint_dir=checkpoint_dir, checkpoint_every=CHECKPOINT_EVERY,
        device=DEVICE, class_weights=class_weights,
    )

    # Extract hidden states for later analysis (PCA, stability, etc.)
    print("\nExtracting hidden states from trained model...")
    hidden_states = extract_hidden_states(model, X_val[:50], device=DEVICE)
    print(f"Hidden states shape: {hidden_states.shape}")
    np.save(os.path.join(checkpoint_dir, 'hidden_states_val.npy'), hidden_states)
    print(f"Saved to {checkpoint_dir}/hidden_states_val.npy")

    # Save training history for plotting learning curves
    np.savez(
        os.path.join(checkpoint_dir, 'training_history.npz'),
        train_loss=history['train_loss'], train_acc=history['train_acc'],
        train_dec_acc=history['train_dec_acc'], val_loss=history['val_loss'],
        val_acc=history['val_acc'], val_dec_acc=history['val_dec_acc'],
    )
    print("Done! Training history saved.")


if __name__ == '__main__':
    main()
