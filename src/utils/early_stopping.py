"""Early Stopping and Model Checkpoint utilities."""
import os
import torch


class EarlyStopping:
    """Early stops training if monitored validation metric doesn't improve after a given patience."""

    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 0.01,
        mode: str = "min",
        checkpoint_path: str = "checkpoints/best_egsq_multitask.pth",
        warmup_epochs: int = 15,
        verbose: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode.lower()
        self.checkpoint_path = checkpoint_path
        self.warmup_epochs = warmup_epochs
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

        assert self.mode in ["min", "max"], "mode must be 'min' or 'max'"

    def __call__(self, val_metric: float, model: torch.nn.Module, epoch: int) -> bool:
        if self.mode == "min":
            score = -val_metric
        else:
            score = val_metric

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(val_metric, model, epoch)
            return False

        if score < self.best_score + self.min_delta:
            if epoch > self.warmup_epochs:
                self.counter += 1
                if self.verbose:
                    print(f"EarlyStopping counter: {self.counter} out of {self.patience} (Best: {-self.best_score if self.mode == 'min' else self.best_score:.4f} at epoch {self.best_epoch})")
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                if self.verbose:
                    print(f"Epoch {epoch} <= Warmup ({self.warmup_epochs}): EarlyStopping counter held at 0.")
        else:
            if self.verbose:
                print(f"Validation metric improved ({-self.best_score if self.mode == 'min' else self.best_score:.4f} -> {val_metric:.4f}). Saving model...")
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(val_metric, model, epoch)
            self.counter = 0

        return self.early_stop

    def save_checkpoint(self, val_metric: float, model: torch.nn.Module, epoch: int):
        """Saves model when validation metric improves."""
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metric_value": val_metric,
            "metric_mode": self.mode,
        }, self.checkpoint_path)
        if self.verbose:
            print(f"Saved checkpoint to {self.checkpoint_path}")
