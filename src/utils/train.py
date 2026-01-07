import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Tuple, Optional

from ..models.transformer_variants import create_transformer_model
from ..models.lstm import create_lstm_model
from ..models.nbeats import create_nbeats_model


MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def create_model(model_type: str, 
                 input_dim: int, 
                 window_size: int,
                 horizon: int = 1,
                 **kwargs) -> nn.Module:
    kwargs.pop('target_scaler', None)
    
    if model_type in ['transformer_encoder', 'transformer_decoder', 'vanilla_transformer']:
        return create_transformer_model(model_type, input_dim, horizon, **kwargs)
    elif model_type == 'lstm':
        return create_lstm_model(input_dim, horizon, **kwargs)
    elif model_type == 'nbeats':
        return create_nbeats_model(input_dim, window_size, horizon, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Metrics
def compute_mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean(np.abs(predictions - targets)))


def compute_mse(predictions: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean((predictions - targets) ** 2))


def compute_correlation(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute Pearson correlation."""
    if len(predictions) < 2:
        return 0.0
    pred_flat = predictions.flatten()
    tgt_flat = targets.flatten()
    if pred_flat.std() < 1e-8 or tgt_flat.std() < 1e-8:
        return 0.0
    corr = np.corrcoef(pred_flat, tgt_flat)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    return {
        'mae': compute_mae(predictions, targets),
        'mse': compute_mse(predictions, targets),
        'corr': compute_correlation(predictions, targets)
    }


class DirectionalVarianceLoss(nn.Module):
    """
    Combines MSE, Directional Penalty, and Variance Regularization.
    """
    def __init__(self, mse_weight: float = 1.0, direction_weight: float = 1.0, var_weight: float = 0.0):
        super().__init__()
        self.mse_weight = mse_weight
        self.direction_weight = direction_weight
        self.var_weight = var_weight
        self.mse = nn.MSELoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 1. MSE Loss
        mse_loss = self.mse(pred, target)
        
        # 2. Directional Penalty
        direction_loss = torch.mean(torch.relu(-pred * target))
        
        # 3. Variance Penalty (Encourage variance)
        pred_var = torch.var(pred, unbiased=False)
        target_var = torch.var(target, unbiased=False) + 1e-6
        var_ratio = pred_var / target_var
        var_loss = torch.abs(var_ratio - 1.0)
        
        total_loss = (self.mse_weight * mse_loss + 
                      self.direction_weight * direction_loss + 
                      self.var_weight * var_loss)
        return total_loss


def train_single_model(model: nn.Module,
                       train_loader: DataLoader,
                       val_loader: DataLoader,
                       num_epochs: int = 100,
                       lr: float = 1e-3,
                       patience: int = 25,
                       device: str = 'auto',
                       verbose: bool = True,
                       loss_type: str = 'directional') -> Tuple[nn.Module, Dict]:
    """
    Train model with choice of loss function.
    
    Args:
        loss_type: 'mse' for pure MSE, 'directional' for custom loss
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = model.to(device)
    
    # Choose loss function
    if loss_type == 'mse':
        criterion = nn.MSELoss()
    else:
        # Directional Loss to encourage correct sign
        # High direction_weight to force model to learn trend
        criterion = DirectionalVarianceLoss(
            mse_weight=1.0, 
            direction_weight=5.0, 
            var_weight=0.1
        )
    
    # AdamW with OneCycleLR for better convergence
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=num_epochs, steps_per_epoch=len(train_loader)
    )
    
    history = {'train_loss': [], 'val_loss': [], 'val_mae': [], 'val_corr': [], 'val_acc': []}
    
    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            
            if outputs.dim() > 1:
                outputs = outputs.squeeze(-1)
            
            loss = criterion(outputs, y_batch)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                outputs = model(X_batch)
                if outputs.dim() > 1:
                    outputs = outputs.squeeze(-1)
                
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                
                val_preds.append(outputs.cpu().numpy())
                val_targets.append(y_batch.cpu().numpy())
        
        val_loss /= len(val_loader)
        history['val_loss'].append(val_loss)
        
        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        
        val_mae = compute_mae(val_preds, val_targets)
        val_corr = compute_correlation(val_preds, val_targets)
        
        history['val_mae'].append(val_mae)
        history['val_corr'].append(val_corr)
        
        if verbose:
            print(f"  Epoch [{epoch+1}/{num_epochs}] "
                  f"Loss: {train_loss:.4f} | "
                  f"V.Loss: {val_loss:.4f} | "
                  f"MAE: {val_mae:.4f} | "
                  f"Corr: {val_corr:.3f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history

def save_model(model: nn.Module, model_type: str, window_size: int, horizon: int, 
               extra_info: str = "") -> str:
    """Save model to models/ directory."""
    filename = f"{model_type}_w{window_size}_h{horizon}"
    if extra_info:
        filename += f"_{extra_info}"
    filename += ".pth"
    
    filepath = os.path.join(MODELS_DIR, filename)
    torch.save(model.state_dict(), filepath)
    return filepath


def load_model(model_type: str, input_dim: int, window_size: int, horizon: int,
               extra_info: str = "") -> nn.Module:
    """Load model from models/ directory."""
    filename = f"{model_type}_w{window_size}_h{horizon}"
    if extra_info:
        filename += f"_{extra_info}"
    filename += ".pth"
    
    filepath = os.path.join(MODELS_DIR, filename)
    
    model = create_model(model_type, input_dim, window_size, horizon)
    model.load_state_dict(torch.load(filepath, map_location='cpu'))
    return model

def evaluate_model(model: nn.Module,
                   test_loader: DataLoader,
                   device: str = 'auto') -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate model on test data."""
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            
            if outputs.dim() > 1:
                outputs = outputs.squeeze(-1)
            
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(y_batch.numpy())
    
    predictions = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    
    metrics = compute_metrics(predictions, targets)
    return metrics, predictions, targets

def train_and_evaluate_per_ticker(model_type: str,
                                  X_train: np.ndarray,
                                  y_train: np.ndarray,
                                  X_val: np.ndarray,
                                  y_val: np.ndarray,
                                  test_per_ticker: Dict[str, Tuple[np.ndarray, np.ndarray]],
                                  window_size: int,
                                  horizon: int,
                                  batch_size: int = 32,
                                  num_epochs: int = 100,
                                  lr: float = 5e-4,
                                  patience: int = 20,
                                  verbose: bool = True,
                                  save_best: bool = True,
                                  **model_kwargs) -> Dict:
    from .data_utils import StockDataset
    
    input_dim = X_train.shape[-1]
    
    # Create datasets
    train_dataset = StockDataset(X_train, y_train)
    val_dataset = StockDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create and train model
    model = create_model(model_type, input_dim, window_size, horizon, **model_kwargs)
    
    model, history = train_single_model(
        model, train_loader, val_loader,
        num_epochs=num_epochs,
        lr=lr,
        patience=patience,
        verbose=verbose
    )
    
    # Save best model
    model_path = None
    if save_best:
        model_path = save_model(model, model_type, window_size, horizon)
        if verbose:
            print(f"  Model saved: {model_path}")
    
    # Evaluate per ticker
    results_per_ticker = {}
    
    # Get target_scaler for inverse transform (Log Price -> Price)
    target_scaler = model_kwargs.get('target_scaler', None)
    
    for ticker, test_data in test_per_ticker.items():
        X_test = test_data[0]
        y_test_scaled = test_data[1]
        dates_test = test_data[2] if len(test_data) > 2 else None
        y_test_log = test_data[3] if len(test_data) > 3 else None
        
        test_dataset = StockDataset(X_test, y_test_scaled)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        metrics, predictions_scaled, targets_scaled = evaluate_model(model, test_loader)
        
        if target_scaler is not None and y_test_log is not None:
            pred_log = target_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
            predictions_plot = np.exp(pred_log)
            targets_plot = np.exp(y_test_log)
        else:
            predictions_plot = predictions_scaled
            targets_plot = targets_scaled
        
        results_per_ticker[ticker] = {
            'mae': metrics['mae'],
            'mse': metrics['mse'],
            'corr': metrics['corr'],
            'predictions': predictions_plot,
            'targets': targets_plot
        }
        
        if verbose:
            print(f"  {ticker} - MAE: {metrics['mae']:.4f}, MSE: {metrics['mse']:.4f}, Corr: {metrics['corr']:.3f}")
    
    return {
        'model': model,
        'results_per_ticker': results_per_ticker,
        'model_path': model_path,
        'history': history
    }
