import numpy as np
import torch
import torch.nn as nn

from .base import BaseAnomalyDetector


class _LSTMAE(nn.Module):
    def __init__(self, input_size=1, hidden=32, layers=2):
        super().__init__()
        self.enc = nn.LSTM(input_size, hidden, layers, batch_first=True)
        self.dec = nn.LSTM(hidden, hidden, layers, batch_first=True)
        self.out = nn.Linear(hidden, input_size)

    def forward(self, x):
        _, (h, c) = self.enc(x)
        z = h[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        y, _ = self.dec(z, (h, c))
        return self.out(y)


class LSTMAutoencoderDetector(BaseAnomalyDetector):
    """Operates on raw sequences: input shape = [seq_len, 1] flattened to [seq_len]."""
    name = "lstm_autoencoder"

    def __init__(self, seq_len=32, hidden=32, layers=2, epochs=40,
                 batch_size=64, lr=1e-3, device="cpu"):
        self.seq_len = seq_len
        self.hidden = hidden
        self.layers = layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device(device)
        self.model: _LSTMAE | None = None
        self.err_stats = (0.0, 1.0)

    def _to_seq(self, x: np.ndarray) -> np.ndarray:
        """x: [N, seq_len] → [N, seq_len, 1]"""
        return x.reshape(x.shape[0], self.seq_len, 1).astype(np.float32)

    def fit(self, X: np.ndarray) -> "LSTMAutoencoderDetector":
        Xs = self._to_seq(X)
        self.model = _LSTMAE(1, self.hidden, self.layers).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        X_t = torch.tensor(Xs, dtype=torch.float32)
        dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_t), batch_size=self.batch_size, shuffle=True
        )

        self.model.train()
        for epoch in range(self.epochs):
            total = 0.0
            for (b,) in dl:
                b = b.to(self.device)
                opt.zero_grad()
                out = self.model(b)
                loss = loss_fn(out, b)
                loss.backward()
                opt.step()
                total += loss.item() * b.size(0)
            if (epoch + 1) % 10 == 0:
                print(f"[LSTM-AE] epoch {epoch+1}/{self.epochs} loss={total/len(X_t):.6f}")

        self.model.eval()
        with torch.no_grad():
            errs = ((self.model(X_t.to(self.device)) - X_t.to(self.device)) ** 2).mean(dim=(1, 2)).cpu().numpy()
        self.err_stats = (float(errs.mean()), float(errs.std() + 1e-8))
        return self

    def score(self, x: np.ndarray) -> float:
        x = x.reshape(1, self.seq_len, 1).astype(np.float32)
        self.model.eval()
        with torch.no_grad():
            xt = torch.tensor(x).to(self.device)
            err = float(((self.model(xt) - xt) ** 2).mean().item())
        m, s = self.err_stats
        z = (err - m) / s
        return float(1.0 / (1.0 + np.exp(-(z - 3.0))))