import numpy as np
import torch
import torch.nn as nn

from .base import BaseAnomalyDetector


class _AE(nn.Module):
    def __init__(self, dim: int, hidden: list[int]):
        super().__init__()
        enc_layers = []
        prev = dim
        for h in hidden:
            enc_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.encoder = nn.Sequential(*enc_layers)
        dec_layers = []
        for h in reversed(hidden[:-1] + [dim]):
            dec_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class AutoencoderDetector(BaseAnomalyDetector):
    name = "autoencoder"

    def __init__(self, hidden_dims=(64, 16, 64), epochs=40, batch_size=128,
                 lr=1e-3, device="cpu"):
        self.hidden_dims = list(hidden_dims)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device(device)
        self.model: _AE | None = None
        self.error_stats: tuple[float, float] = (0.0, 1.0)

    def fit(self, X: np.ndarray) -> "AutoencoderDetector":
        dim = X.shape[1]
        self.model = _AE(dim, self.hidden_dims).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        X_t = torch.tensor(X, dtype=torch.float32)
        ds = torch.utils.data.TensorDataset(X_t)
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            total = 0.0
            for (batch,) in dl:
                batch = batch.to(self.device)
                opt.zero_grad()
                out = self.model(batch)
                loss = loss_fn(out, batch)
                loss.backward()
                opt.step()
                total += loss.item() * batch.size(0)
            if (epoch + 1) % 10 == 0:
                print(f"[Autoencoder] epoch {epoch+1}/{self.epochs} loss={total/len(X_t):.6f}")

        # calibrate errors
        self.model.eval()
        with torch.no_grad():
            errs = ((self.model(X_t.to(self.device)) - X_t.to(self.device)) ** 2).mean(dim=1).cpu().numpy()
        self.error_stats = (float(errs.mean()), float(errs.std() + 1e-8))
        return self

    def _recon_error(self, x: np.ndarray) -> float:
        self.model.eval()
        with torch.no_grad():
            xt = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device)
            out = self.model(xt)
            return float(((out - xt) ** 2).mean().item())

    def score(self, x: np.ndarray) -> float:
        err = self._recon_error(x)
        m, s = self.error_stats
        z = (err - m) / s
        return float(1.0 / (1.0 + np.exp(-(z - 3.0))))