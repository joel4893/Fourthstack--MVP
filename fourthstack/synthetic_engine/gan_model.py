import torch
import torch.nn as nn
from typing import Optional, Any, Iterable


class Generator(nn.Module):
    def __init__(self, input_dim: int = 64, output_dim: int = 32) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim: int = 32) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def sample_noise(batch_size: int, dim: int = 64) -> torch.Tensor:
    return torch.randn(batch_size, dim)


class GANModel:
    """Simple GAN wrapper with minimal train/generate functionality.

    This is a lightweight helper meant for testing pipelines; it is not
    intended as a production GAN implementation.
    """

    def __init__(
        self, input_dim: int = 64, output_dim: int = 32, device: Optional[Any] = None
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gen = Generator(input_dim, output_dim).to(self.device)
        self.disc = Discriminator(output_dim).to(self.device)
        self.input_dim = input_dim

    def train(
        self, data: Iterable[torch.Tensor], epochs: int = 1, lr: float = 1e-4
    ) -> bool:
        """Minimal training loop. `data` should be an iterable of torch tensors.

        This trains for a small number of steps and is intended only for smoke tests.
        """
        gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr)
        disc_opt = torch.optim.Adam(self.disc.parameters(), lr=lr)
        loss_fn = nn.BCELoss()

        for _ in range(epochs):
            for real in data:
                real = real.to(self.device).float()
                batch_size = real.size(0)

                # Train discriminator
                disc_opt.zero_grad()
                real_labels = torch.ones(batch_size, 1, device=self.device)
                real_pred = self.disc(real)
                loss_real = loss_fn(real_pred, real_labels)

                noise = sample_noise(batch_size, self.input_dim).to(self.device)
                fake = self.gen(noise)
                fake_labels = torch.zeros(batch_size, 1, device=self.device)
                fake_pred = self.disc(fake.detach())
                loss_fake = loss_fn(fake_pred, fake_labels)

                loss_disc = (loss_real + loss_fake) * 0.5
                loss_disc.backward()
                disc_opt.step()

                # Train generator
                gen_opt.zero_grad()
                pred = self.disc(fake)
                loss_gen = loss_fn(pred, real_labels)
                loss_gen.backward()
                gen_opt.step()

        return True

    def generate(self, n: int) -> list:
        """Generate `n` synthetic samples and return as a list.

        Returns a list of lists (rows) representing the raw generator output.
        """
        self.gen.eval()
        with torch.no_grad():
            noise = sample_noise(n, self.input_dim).to(self.device)
            out = self.gen(noise).cpu()
        return out.numpy().tolist()
