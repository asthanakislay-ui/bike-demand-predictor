import torch
import torch.nn as nn
import numpy as np
from bike_demand_predictor.config import DEFAULT_LR, DEFAULT_BATCH_SIZE, HIDDEN_LAYERS

class BikeDemandANN(nn.Module):
    """
    Artificial Neural Network for Bike Demand Prediction.
    The architecture is parameterized via hidden_layers for flexibility.
    """
    def __init__(self, input_features, hidden_layers=[64, 32]):
        super().__init__()

        layers = []
        in_dim = input_features

        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def train_model(model, train_loader, val_loader, epochs=100, patience=10):
    """
    Trains the model with Early Stopping to prevent overfitting.

    Args:
        model: The PyTorch model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: Maximum number of epochs
        patience: Number of epochs to wait for improvement before stopping

    Returns:
        model: The model with the best weights restored
        train_losses: List of average training loss per epoch
        val_losses: List of average validation loss per epoch
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=DEFAULT_LR)

    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        batch_losses = []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        avg_train_loss = np.mean(batch_losses)
        train_losses.append(avg_train_loss)

        # --- Validation Phase ---
        model.eval()
        val_batch_losses = []
        with torch.no_grad():
            for X_val, y_val in val_loader:
                preds_val = model(X_val)
                v_loss = criterion(preds_val, y_val)
                val_batch_losses.append(v_loss.item())

        avg_val_loss = np.mean(val_batch_losses)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # --- Early Stopping Logic ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict() # Save the best version
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}!")
                break

    # Restore best weights to ensure we don't return an overfitted model
    if best_model_state:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses
