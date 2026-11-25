"""Neural paradigm: autoencoder + FCM (two-stage).

An MLP autoencoder (d -> 128 -> 64 -> z) is trained with mean-squared reconstruction error;
FCM is then run on the standardized latent codes. The outlier score is the reconstruction error.
Set denoise > 0 for a denoising autoencoder (Gaussian input corruption, Vincent et al. 2008).

Training length: if `steps` is given, the number of epochs is chosen so that training makes about
that many mini-batch updates (epochs = ceil(steps * batch_size / n)). This keeps the training budget,
and the runtime, comparable across datasets of very different sizes. Otherwise `epochs` is used.
"""
import numpy as np
from .base import FitResult
from .fcm import fcm_best


class AEFCM:
    name = "AE+FCM"
    paradigm, mechanism = "neural/hybrid", "M4 regularized structure"

    def __init__(self, latent=None, epochs=150, batch_size=64, lr=1e-3, denoise=0.0,
                 m=2.0, n_init=5, seed=0, threads=1, steps=None):
        self.latent, self.epochs, self.batch_size, self.lr = latent, epochs, batch_size, lr
        self.steps = steps
        self.denoise, self.m, self.n_init, self.seed, self.threads = denoise, m, n_init, seed, threads

    def fit(self, X, c):
        try:
            import torch
            from torch import nn
        except ImportError as e:  # pragma: no cover
            raise ImportError("AE+FCM needs PyTorch: pip install torch (see README)") from e
        torch.manual_seed(self.seed)
        torch.set_num_threads(self.threads)
        n, d = X.shape
        z = self.latent or int(min(10, max(2, d // 2)))
        enc = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, z))
        dec = nn.Sequential(nn.Linear(z, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, d))
        params = list(enc.parameters()) + list(dec.parameters())
        opt = torch.optim.Adam(params, lr=self.lr)
        Xt = torch.tensor(X, dtype=torch.float32)
        g = torch.Generator().manual_seed(self.seed)
        losses = []
        epochs = self.epochs if not self.steps else max(1, int(np.ceil(self.steps * self.batch_size / n)))
        for _ in range(epochs):
            perm = torch.randperm(n, generator=g)
            ep = 0.0
            for i in range(0, n, self.batch_size):
                xb = Xt[perm[i:i + self.batch_size]]
                xin = xb + self.denoise * torch.randn(xb.shape, generator=g) if self.denoise > 0 else xb
                loss = ((dec(enc(xin)) - xb) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                ep += float(loss.item()) * len(xb)
            losses.append(ep / n)
        with torch.no_grad():
            Z = enc(Xt)
            rec = ((dec(Z) - Xt) ** 2).mean(1).numpy()
            Z = Z.numpy().astype(float)
        Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-8)
        rng = np.random.default_rng(self.seed)
        V, U, D2, J, it, hist = fcm_best(Z, c, self.m, self.n_init, rng)
        return FitResult(U.argmax(0), rec, None,
                         {"latent_dim": z, "final_loss": losses[-1], "loss_history": losses, "Z": Z, "U": U,
                          "centers": V, "iterations": it, "history": hist})
