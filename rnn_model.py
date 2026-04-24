"""
Vanilla continuous-time RNN for the Poisson-distractor DMS task.

Extracted verbatim from wm_task.ipynb so training/fine-tuning scripts can import
VanillaRNN, TrialDataset, train_model, evaluate, and extract_hidden_states as a
proper module (run_sweep.py already expected this file).
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class TrialDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class VanillaRNN(nn.Module):
    """Continuous-time leaky-integrator RNN (Yang et al. 2019):
        h_{t+1} = (1 - alpha) * h_t + alpha * tanh(W_in x + W_rec h)
    with alpha = dt/tau.
    """

    def __init__(self, input_size=33, hidden_size=128, output_size=3,
                 dt=20, tau=500, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.alpha = dt / tau

        self.W_in = nn.Linear(input_size, hidden_size)
        self.W_rec = nn.Linear(hidden_size, hidden_size, bias=False)
        self.readout = nn.Linear(hidden_size, output_size)

        nn.init.orthogonal_(self.W_rec.weight, gain=1.0)
        nn.init.xavier_uniform_(self.W_in.weight)
        nn.init.xavier_uniform_(self.readout.weight)

    def forward(self, x, h0=None):
        batch_size, seq_len, _ = x.shape
        if h0 is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h = h0.squeeze(0)

        hidden_states = []
        for t in range(seq_len):
            h = (1 - self.alpha) * h + self.alpha * torch.tanh(
                self.W_in(x[:, t]) + self.W_rec(h)
            )
            hidden_states.append(h)

        hidden_states = torch.stack(hidden_states, dim=1)
        predictions = self.readout(hidden_states)
        return predictions, hidden_states


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

            is_decision = (labels_batch >= 1)
            if is_decision.any():
                dec_correct += (predicted_classes[is_decision] == labels_batch[is_decision]).sum().item()
                dec_count += is_decision.sum().item()

    avg_loss = total_loss / len(loader.dataset)
    accuracy = total_correct / total_count
    dec_accuracy = dec_correct / dec_count if dec_count > 0 else 0.0
    return avg_loss, accuracy, dec_accuracy


def train_model(model, train_loader, val_loader=None, num_epochs=50,
                learning_rate=1e-3, checkpoint_dir='checkpoints',
                checkpoint_every=10, device='cpu', class_weights=None,
                log_prefix=''):
    os.makedirs(checkpoint_dir, exist_ok=True)
    model = model.to(device)

    loss_function = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        'train_loss': [], 'train_acc': [], 'train_dec_acc': [],
        'val_loss': [], 'val_acc': [], 'val_dec_acc': [],
    }

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        dec_correct = 0
        dec_total = 0

        for inputs_batch, labels_batch in train_loader:
            inputs_batch = inputs_batch.to(device)
            labels_batch = labels_batch.to(device)

            predictions, _ = model(inputs_batch)
            loss = loss_function(
                predictions.reshape(-1, predictions.size(-1)),
                labels_batch.reshape(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * inputs_batch.size(0)
            predicted_classes = predictions.argmax(dim=-1)
            correct += (predicted_classes == labels_batch).sum().item()
            total += labels_batch.numel()

            is_decision = (labels_batch >= 1)
            if is_decision.any():
                dec_correct += (predicted_classes[is_decision] == labels_batch[is_decision]).sum().item()
                dec_total += is_decision.sum().item()

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct / total
        train_dec_acc = dec_correct / dec_total if dec_total > 0 else 0.0

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_dec_acc'].append(train_dec_acc)

        val_loss, val_acc, val_dec_acc = None, None, None
        if val_loader is not None:
            val_loss, val_acc, val_dec_acc = evaluate(
                model, val_loader, loss_function, device
            )
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_dec_acc'].append(val_dec_acc)

        msg = f"{log_prefix}Epoch {epoch:3d}/{num_epochs} | "
        msg += f"Loss: {train_loss:.4f}  DecAcc: {train_dec_acc:.4f}"
        if val_loss is not None:
            msg += f" | ValLoss: {val_loss:.4f}  ValDecAcc: {val_dec_acc:.4f}"
        print(msg, flush=True)

        if epoch % checkpoint_every == 0 or epoch == num_epochs:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'W_hh': model.W_rec.weight.detach().cpu().numpy(),
                'W_ih': model.W_in.weight.detach().cpu().numpy(),
            }
            path = os.path.join(checkpoint_dir, f'epoch_{epoch:03d}.pt')
            torch.save(checkpoint, path)

    return history


def extract_hidden_states(model, X, device='cpu'):
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        _, hidden_states = model(X_tensor)
    return hidden_states.cpu().numpy()
