import torch
from torch.utils.data import DataLoader
from typing import Optional


class Trainer:
        """Tiny trainer scaffold for training PyTorch models.

        This trainer assumes the dataloader yields input tensors (for
        autoencoder-style reconstruction tasks where target == input).
        """

        def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss_fn, device: str = "cpu"):
                self.device = torch.device(device)
                self.model = model.to(self.device)
                self.optimizer = optimizer
                self.loss_fn = loss_fn

        def train_step(self, batch: torch.Tensor) -> float:
                """Run a single training step and return the loss value."""
                self.model.train()
                self.optimizer.zero_grad()

                inputs = batch.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, inputs)

                loss.backward()
                self.optimizer.step()

                return float(loss.item())

        def train(self, dataloader: DataLoader, epochs: int = 5, log_interval: Optional[int] = None):
                """Train for a number of epochs. Prints progress by epoch.

                Args:
                        dataloader: DataLoader yielding input tensors
                        epochs: number of epochs to run
                        log_interval: if provided, print a progress line every `log_interval` batches
                """
                for epoch in range(1, epochs + 1):
                        total_loss = 0.0
                        batch_count = 0
                        for i, batch in enumerate(dataloader, start=1):
                                loss = self.train_step(batch)
                                total_loss += loss
                                batch_count += 1

                                if log_interval and (i % log_interval == 0):
                                        print(f"Epoch {epoch} | Batch {i} | Recent loss: {loss:.4f}")

                        avg_loss = total_loss / batch_count if batch_count else 0.0
                        print(f"Epoch {epoch}/{epochs} | Avg Loss: {avg_loss:.4f}")
