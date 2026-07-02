"""
Model registry.

Two interfaces depending on task_type:
  binary classifier:  .fit(X, y) -> self,  .predict_proba(X)  
  regressor:          .fit(X, y) -> self,  .predict(X)       -
Each model class declares its task_type so the experiment runner only
applies it to matching tasks.

Adding a model = one class + one entry in MODELS at the bottom.
"""
from __future__ import annotations
import warnings
import random
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings(
    "ignore", category=FutureWarning,
    message=".*penalty.*was deprecated.*",
)
warnings.filterwarnings(
    "ignore", category=UserWarning,
    message=".*Inconsistent values: penalty.*",
)


# ---------------------------------------------------------------------------
# Base interfaces
# ---------------------------------------------------------------------------
class BaseClassifier:
    name: str = "base_clf"
    task_type: str = "binary"

    def fit(self, X, y):  raise NotImplementedError
    def predict_proba(self, X):  raise NotImplementedError


class BaseRegressor:
    name: str = "base_reg"
    task_type: str = "regression"

    def fit(self, X, y):  raise NotImplementedError
    def predict(self, X):  raise NotImplementedError


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------
class SparseLogistic(BaseClassifier):
    name = "sparse_logistic"
    task_type = "binary"

    def __init__(self, C=1.0, penalty="l1", solver="liblinear",
                 max_iter=2000, random_state=None, **_):
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.max_iter = max_iter
        self.random_state = random_state
        self._scaler = StandardScaler()
        self._model = None

    def fit(self, X, y):
        Xs = self._scaler.fit_transform(X)
        self._model = LogisticRegression(
            C=self.C, penalty=self.penalty, solver=self.solver,
            max_iter=self.max_iter, class_weight="balanced",
            random_state=self.random_state,
        )
        self._model.fit(Xs, y)
        return self

    def predict_proba(self, X):
        Xs = self._scaler.transform(X)
        return self._model.predict_proba(Xs)[:, 1]


class XGBClassifierWrap(BaseClassifier):
    name = "xgb_clf"
    task_type = "binary"

    def __init__(self, n_estimators=500, max_depth=4, learning_rate=0.05,
                 subsample=0.8, colsample_bytree=0.8, random_state=None, **_):
        from xgboost import XGBClassifier
        self._XGBClassifier = XGBClassifier
        self.params = dict(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, subsample=subsample,
            colsample_bytree=colsample_bytree,
            eval_metric="logloss", n_jobs=4, verbosity=1,
            random_state=random_state,
        )
        self._model = None

    def fit(self, X, y):
        n_pos = int(np.sum(y == 1))
        n_neg = int(np.sum(y == 0))
        spw = (n_neg / n_pos) if n_pos > 0 else 1.0
        self._model = self._XGBClassifier(scale_pos_weight=spw, **self.params)
        self._model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self._model.predict_proba(X)[:, 1]




class TorchMLPBase:
    """Small one-hidden-layer MLP using the same core hyperparameters as 07c.

    This is intentionally a plain predictor, not a bottleneck model: there is no
    information bottleneck. The post-ReLU hidden activation can be tapped as a
    deterministic latent via encode_latent() for the latent-space analysis,
    without altering the predictor (the forward path and weights are unchanged).
    Architecture:
        X -> Linear(input, hidden_dim) -> ReLU -> Dropout -> Linear(hidden_dim, 1)
    """
    def _set_seed(self):
        if self.random_state is not None:
            random.seed(int(self.random_state))
            np.random.seed(int(self.random_state))
            try:
                import torch
                torch.manual_seed(int(self.random_state))
                torch.cuda.manual_seed_all(int(self.random_state))
            except Exception:
                pass

    def _make_net(self, x_dim):
        import torch.nn as nn
        return nn.Sequential(
            nn.Linear(x_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )

    def encode_latent(self, X, batch_size=None):
        """Return the post-ReLU hidden activation as a deterministic latent.

        This is the concatenation-MLP counterpart to the two-tower's pair latent:
        the (hidden_dim)-dimensional representation read straight out of the
        trained network. It does not change the predictor; the forward path used
        by predict_proba/predict is untouched.
        """
        import torch
        if batch_size is None:
            batch_size = self.batch_size * 8
        Xs = self._scaler.transform(X).astype(np.float32)
        hs = []
        self._model.eval()
        with torch.no_grad():
            for start in range(0, Xs.shape[0], batch_size):
                xb = torch.as_tensor(Xs[start:start + batch_size], dtype=torch.float32, device=self._device)
                h = self._model[1](self._model[0](xb))
                hs.append(h.cpu().numpy())
        return np.vstack(hs)


class MLPClassifier(BaseClassifier, TorchMLPBase):
    name = "mlp_clf"
    task_type = "binary"

    def __init__(self, hidden_dim=64, dropout=0.3, batch_size=512, lr=3e-4,
                 weight_decay=1e-3, n_epochs=200, random_state=None, **_):
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.n_epochs = int(n_epochs)
        self.random_state = random_state
        self._scaler = StandardScaler()
        self._model = None
        self._device = None

    def fit(self, X, y):
        import torch
        import torch.nn.functional as F
        from torch.utils.data import TensorDataset, DataLoader

        self._set_seed()
        Xs = self._scaler.fit_transform(X).astype(np.float32)
        y = np.asarray(y).astype(np.float32)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_t = torch.as_tensor(Xs, dtype=torch.float32, device=self._device)
        y_t = torch.as_tensor(y, dtype=torch.float32, device=self._device)
        ds = TensorDataset(X_t, y_t)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        self._model = self._make_net(X_t.shape[1]).to(self._device)
        opt = torch.optim.AdamW(self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        pos_frac = float(y_t.mean().item())
        pos_weight = torch.tensor((1.0 - pos_frac) / pos_frac, dtype=torch.float32, device=self._device) if pos_frac > 0 else None

        for _ in range(self.n_epochs):
            self._model.train()
            for xb, yb in loader:
                opt.zero_grad()
                logits = self._model(xb).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight)
                loss.backward()
                opt.step()
        return self

    def predict_proba(self, X):
        import torch
        Xs = self._scaler.transform(X).astype(np.float32)
        X_t = torch.as_tensor(Xs, dtype=torch.float32, device=self._device)
        self._model.eval()
        with torch.no_grad():
            logits = self._model(X_t).squeeze(-1)
            return torch.sigmoid(logits).cpu().numpy()

# ---------------------------------------------------------------------------
# Regressors
# ---------------------------------------------------------------------------
class LinearRegressor(BaseRegressor):
    """
    Ridge regression with standardization.
    alpha=0 reduces to OLS. alpha>0 adds L2 penalty (helps when p > n
    or features are correlated, which is the typical genomic case).
    """
    name = "linreg"
    task_type = "regression"

    def __init__(self, alpha=1.0, random_state=None, **_):
        self.alpha = alpha
        self.random_state = random_state
        self._scaler = StandardScaler()
        self._model = None

    def fit(self, X, y):
        Xs = self._scaler.fit_transform(X)
        # Ridge accepts random_state for the 'sag'/'saga' solvers; for the
        # default solver it's a no-op but harmless to pass.
        self._model = Ridge(alpha=self.alpha, random_state=self.random_state)
        self._model.fit(Xs, y)
        return self

    def predict(self, X):
        Xs = self._scaler.transform(X)
        return self._model.predict(Xs)


class MLPRegressor(BaseRegressor, TorchMLPBase):
    name = "mlp_reg"
    task_type = "regression"

    def __init__(self, hidden_dim=64, dropout=0.3, batch_size=512, lr=3e-4,
                 weight_decay=1e-3, n_epochs=200, random_state=None, **_):
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.n_epochs = int(n_epochs)
        self.random_state = random_state
        self._scaler = StandardScaler()
        self._model = None
        self._device = None

    def fit(self, X, y):
        import torch
        import torch.nn.functional as F
        from torch.utils.data import TensorDataset, DataLoader

        self._set_seed()
        Xs = self._scaler.fit_transform(X).astype(np.float32)
        y = np.asarray(y).astype(np.float32)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_t = torch.as_tensor(Xs, dtype=torch.float32, device=self._device)
        y_t = torch.as_tensor(y, dtype=torch.float32, device=self._device)
        ds = TensorDataset(X_t, y_t)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        self._model = self._make_net(X_t.shape[1]).to(self._device)
        opt = torch.optim.AdamW(self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for _ in range(self.n_epochs):
            self._model.train()
            for xb, yb in loader:
                opt.zero_grad()
                pred = self._model(xb).squeeze(-1)
                loss = F.mse_loss(pred, yb)
                loss.backward()
                opt.step()
        return self

    def predict(self, X):
        import torch
        Xs = self._scaler.transform(X).astype(np.float32)
        X_t = torch.as_tensor(Xs, dtype=torch.float32, device=self._device)
        self._model.eval()
        with torch.no_grad():
            return self._model(X_t).squeeze(-1).cpu().numpy()



# ---------------------------------------------------------------------------
# VIB latent-space MLPs copied in spirit from 07c
# ---------------------------------------------------------------------------
class _VIBMixin:
    """Single stochastic latent channel with dual-ascent KL budget.

    Hyperparameter defaults mirror 07c_train_dual_lagrangian_vib_full.py:
      latent_dim=3, hidden_dim=64, dropout=0.3, batch_size=512,
      lr=3e-4, weight_decay=1e-3, n_epochs=200, warmup_epochs=60,
      beta_init=1e-4, dual_lr=0.01, beta_max=100, m=0.6,
      C_min=0.05, KL_smoothing=0.99.
    """
    def _init_vib_params(self, latent_dim=5, hidden_dim=64, dropout=0.3,
                         batch_size=512, lr=3e-4, weight_decay=1e-3,
                         n_epochs=200, warmup_epochs=60, dual_lr=0.01,
                         beta_init=1e-4, beta_max=100.0, m=0.6,
                         C_min=0.05, use_moving_avg=True,
                         KL_smoothing=0.99, lambda_y=1.0, lambda_w=1.0,
                         random_state=None, **_):
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.n_epochs = int(n_epochs)
        self.warmup_epochs = int(warmup_epochs)
        self.dual_lr = float(dual_lr)
        self.beta_init = float(beta_init)
        self.beta_max = float(beta_max)
        self.m = float(m)
        self.C_min = float(C_min)
        self.use_moving_avg = bool(use_moving_avg)
        self.KL_smoothing = float(KL_smoothing)
        self.lambda_y = float(lambda_y)
        self.lambda_w = float(lambda_w)
        self.random_state = random_state
        self._scaler = StandardScaler()
        self._device = None
        self._model = None
        self._info_trace = []

    def _set_seed(self):
        if self.random_state is not None:
            random.seed(int(self.random_state))
            np.random.seed(int(self.random_state))
            try:
                import torch
                torch.manual_seed(int(self.random_state))
                torch.cuda.manual_seed_all(int(self.random_state))
            except Exception:
                pass

    def _make_encoder(self, x_dim):
        import torch.nn as nn
        return nn.Sequential(
            nn.Linear(x_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.latent_dim * 2),
        )

    @staticmethod
    def _kl_diag_gaussian(mu, logvar):
        import torch
        return 0.5 * torch.mean(torch.sum(torch.exp(logvar) + mu.pow(2) - 1.0 - logvar, dim=1))

    @staticmethod
    def _sample_z(mu, logvar):
        import torch
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def get_info_trace(self):
        return list(self._info_trace)


def _make_vib_y_net(x_dim, latent_dim, hidden_dim, dropout):
    import torch
    import torch.nn as nn
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Linear(x_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim, latent_dim * 2)
            )
            self.head_y = nn.Linear(latent_dim, 1)
        def encode(self, X):
            mu, logvar = torch.chunk(self.enc(X), 2, dim=-1)
            logvar = torch.clamp(logvar, min=-10.0, max=5.0)
            return mu, logvar
        def forward(self, X):
            mu, logvar = self.encode(X)
            return mu, logvar, self.head_y(mu).squeeze(-1)
    return Net()


def _make_vib_w_net(x_dim, latent_dim, hidden_dim, dropout):
    import torch
    import torch.nn as nn
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Linear(x_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim, latent_dim * 2)
            )
            self.head_w = nn.Linear(latent_dim, 1)
        def encode(self, X):
            mu, logvar = torch.chunk(self.enc(X), 2, dim=-1)
            logvar = torch.clamp(logvar, min=-10.0, max=5.0)
            return mu, logvar
        def forward(self, X):
            mu, logvar = self.encode(X)
            return mu, logvar, self.head_w(mu).squeeze(-1)
    return Net()


def _make_vib_yw_net(x_dim, latent_dim, hidden_dim, dropout):
    import torch
    import torch.nn as nn
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Linear(x_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim, latent_dim * 2)
            )
            self.head_y = nn.Linear(latent_dim, 1)
            self.head_w = nn.Linear(latent_dim, 1)
        def encode(self, X):
            mu, logvar = torch.chunk(self.enc(X), 2, dim=-1)
            logvar = torch.clamp(logvar, min=-10.0, max=5.0)
            return mu, logvar
        def forward(self, X):
            mu, logvar = self.encode(X)
            return mu, logvar, self.head_y(mu).squeeze(-1), self.head_w(mu).squeeze(-1)
    return Net()


class VIBYClassifier(BaseClassifier, _VIBMixin):
    name = "vib_y_only"
    task_type = "binary"

    def __init__(self, **params):
        self._init_vib_params(**params)

    def fit(self, X, y):
        import torch
        import torch.nn.functional as F
        from torch.utils.data import TensorDataset, DataLoader

        self._set_seed()
        Xs = self._scaler.fit_transform(X).astype(np.float32)
        y = np.asarray(y).astype(np.float32)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_t = torch.as_tensor(Xs, dtype=torch.float32, device=self._device)
        y_t = torch.as_tensor(y, dtype=torch.float32, device=self._device)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True)
        self._model = _make_vib_y_net(X_t.shape[1], self.latent_dim, self.hidden_dim, self.dropout).to(self._device)
        opt = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        pos_frac = float(y_t.mean().item())
        pos_weight = torch.tensor((1.0 - pos_frac) / pos_frac, dtype=torch.float32, device=self._device) if pos_frac > 0 else None
        w_pos_frac = float(w_t.mean().item())
        w_pos_weight = torch.tensor((1.0 - w_pos_frac) / w_pos_frac, dtype=torch.float32, device=self._device) if 0.0 < w_pos_frac < 1.0 else None

        beta = self.beta_init
        C = self.C_min
        KL_ma = 0.0
        store_KL = []
        self._info_trace = []
        for epoch in range(1, self.n_epochs + 1):
            n_batches = 0; sum_loss_y = 0.0; sum_KL = 0.0
            self._model.train()
            for xb, yb in loader:
                opt.zero_grad()
                mu, logvar = self._model.encode(xb)
                z = self._sample_z(mu, logvar)
                logits = self._model.head_y(z).squeeze(-1)
                loss_y = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight)
                KL_batch = self._kl_diag_gaussian(mu, logvar)
                if epoch <= self.warmup_epochs:
                    loss = loss_y + beta * KL_batch
                    if epoch > self.warmup_epochs - 10:
                        store_KL.append(KL_batch.item())
                else:
                    if epoch == self.warmup_epochs + 1 and C == self.C_min:
                        KL_med = float(np.median(store_KL)) if store_KL else 1.0
                        C = max(self.C_min, self.m * KL_med)
                        KL_ma = KL_med
                    KL_ma = self.KL_smoothing * KL_ma + (1 - self.KL_smoothing) * KL_batch.item() if self.use_moving_avg else KL_batch.item()
                    loss = loss_y + beta * KL_batch
                loss.backward(); opt.step()
                if epoch > self.warmup_epochs:
                    beta = min(self.beta_max, max(0.0, beta + self.dual_lr * (KL_ma - C)))
                n_batches += 1; sum_loss_y += float(loss_y.item()); sum_KL += float(KL_batch.item())
            self._info_trace.append({
                "epoch": epoch, "phase": "warmup" if epoch <= self.warmup_epochs else "dual",
                "objective": "Y_only", "latent_dim": self.latent_dim,
                "loss_task_mean": sum_loss_y / max(n_batches, 1),
                "loss_y_mean": sum_loss_y / max(n_batches, 1), "loss_w_mean": np.nan,
                "KL_mean": sum_KL / max(n_batches, 1), "beta": float(beta), "C": float(C),
            })
        return self

    def predict_proba(self, X):
        import torch
        Xs = self._scaler.transform(X).astype(np.float32)
        X_t = torch.as_tensor(Xs, dtype=torch.float32, device=self._device)
        self._model.eval()
        with torch.no_grad():
            _, _, logits = self._model(X_t)
            return torch.sigmoid(logits).cpu().numpy()


class VIBWRegressor(BaseRegressor, _VIBMixin):
    name = "vib_w_only"
    task_type = "regression"

    def __init__(self, **params):
        self._init_vib_params(**params)

    def fit(self, X, y):
        import torch
        import torch.nn.functional as F
        from torch.utils.data import TensorDataset, DataLoader

        self._set_seed()
        Xs = self._scaler.fit_transform(X).astype(np.float32)
        y = np.asarray(y).astype(np.float32)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_t = torch.as_tensor(Xs, dtype=torch.float32, device=self._device)
        y_t = torch.as_tensor(y, dtype=torch.float32, device=self._device)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True)
        self._model = _make_vib_w_net(X_t.shape[1], self.latent_dim, self.hidden_dim, self.dropout).to(self._device)
        opt = torch.optim.Adam(self._model.parameters(), lr=self.lr)

        beta = self.beta_init
        C = self.C_min
        KL_ma = 0.0
        store_KL = []
        self._info_trace = []
        for epoch in range(1, self.n_epochs + 1):
            n_batches = 0; sum_loss_w = 0.0; sum_KL = 0.0
            self._model.train()
            for xb, yb in loader:
                opt.zero_grad()
                mu, logvar = self._model.encode(xb)
                z = self._sample_z(mu, logvar)
                pred = self._model.head_w(z).squeeze(-1)
                loss_w = F.mse_loss(pred, yb)
                KL_batch = self._kl_diag_gaussian(mu, logvar)
                if epoch <= self.warmup_epochs:
                    loss = loss_w + beta * KL_batch
                    if epoch > self.warmup_epochs - 10:
                        store_KL.append(KL_batch.item())
                else:
                    if epoch == self.warmup_epochs + 1 and C == self.C_min:
                        KL_med = float(np.median(store_KL)) if store_KL else 1.0
                        C = max(self.C_min, self.m * KL_med)
                        KL_ma = KL_med
                    KL_ma = self.KL_smoothing * KL_ma + (1 - self.KL_smoothing) * KL_batch.item() if self.use_moving_avg else KL_batch.item()
                    loss = loss_w + beta * KL_batch
                loss.backward(); opt.step()
                if epoch > self.warmup_epochs:
                    beta = min(self.beta_max, max(0.0, beta + self.dual_lr * (KL_ma - C)))
                n_batches += 1; sum_loss_w += float(loss_w.item()); sum_KL += float(KL_batch.item())
            self._info_trace.append({
                "epoch": epoch, "phase": "warmup" if epoch <= self.warmup_epochs else "dual",
                "objective": "W_only", "latent_dim": self.latent_dim,
                "loss_task_mean": sum_loss_w / max(n_batches, 1),
                "loss_y_mean": np.nan, "loss_w_mean": sum_loss_w / max(n_batches, 1),
                "KL_mean": sum_KL / max(n_batches, 1), "beta": float(beta), "C": float(C),
            })
        return self

    def predict(self, X):
        import torch
        Xs = self._scaler.transform(X).astype(np.float32)
        X_t = torch.as_tensor(Xs, dtype=torch.float32, device=self._device)
        self._model.eval()
        with torch.no_grad():
            _, _, pred = self._model(X_t)
            return pred.cpu().numpy()


class VIBYWJoint(_VIBMixin):
    name = "vib_yw_joint"
    task_type = "joint_y_w"

    def __init__(self, **params):
        self._init_vib_params(**params)

    def fit(self, X, y_binary, w_continuous):
        import torch
        import torch.nn.functional as F
        from torch.utils.data import TensorDataset, DataLoader

        self._set_seed()
        Xs = self._scaler.fit_transform(X).astype(np.float32)
        y_binary = np.asarray(y_binary).astype(np.float32)
        w_continuous = np.asarray(w_continuous).astype(np.float32)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_t = torch.as_tensor(Xs, dtype=torch.float32, device=self._device)
        y_t = torch.as_tensor(y_binary, dtype=torch.float32, device=self._device)
        w_t = torch.as_tensor(w_continuous, dtype=torch.float32, device=self._device)
        loader = DataLoader(TensorDataset(X_t, y_t, w_t), batch_size=self.batch_size, shuffle=True)
        self._model = _make_vib_yw_net(X_t.shape[1], self.latent_dim, self.hidden_dim, self.dropout).to(self._device)
        opt = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        pos_frac = float(y_t.mean().item())
        pos_weight = torch.tensor((1.0 - pos_frac) / pos_frac, dtype=torch.float32, device=self._device) if pos_frac > 0 else None

        beta = self.beta_init
        C = self.C_min
        KL_ma = 0.0
        store_KL = []
        self._info_trace = []
        for epoch in range(1, self.n_epochs + 1):
            n_batches = 0; sum_loss_task = 0.0; sum_loss_y = 0.0; sum_loss_w = 0.0; sum_KL = 0.0
            self._model.train()
            for xb, yb, wb in loader:
                opt.zero_grad()
                mu, logvar = self._model.encode(xb)
                z = self._sample_z(mu, logvar)
                logits_y = self._model.head_y(z).squeeze(-1)
                pred_w = self._model.head_w(z).squeeze(-1)
                loss_y = F.binary_cross_entropy_with_logits(logits_y, yb, pos_weight=pos_weight)
                loss_w = F.binary_cross_entropy_with_logits(pred_w, wb, pos_weight=w_pos_weight)
                loss_task = self.lambda_y * loss_y + self.lambda_w * loss_w
                KL_batch = self._kl_diag_gaussian(mu, logvar)
                if epoch <= self.warmup_epochs:
                    loss = loss_task + beta * KL_batch
                    if epoch > self.warmup_epochs - 10:
                        store_KL.append(KL_batch.item())
                else:
                    if epoch == self.warmup_epochs + 1 and C == self.C_min:
                        KL_med = float(np.median(store_KL)) if store_KL else 1.0
                        C = max(self.C_min, self.m * KL_med)
                        KL_ma = KL_med
                    KL_ma = self.KL_smoothing * KL_ma + (1 - self.KL_smoothing) * KL_batch.item() if self.use_moving_avg else KL_batch.item()
                    loss = loss_task + beta * KL_batch
                loss.backward(); opt.step()
                if epoch > self.warmup_epochs:
                    beta = min(self.beta_max, max(0.0, beta + self.dual_lr * (KL_ma - C)))
                n_batches += 1
                sum_loss_task += float(loss_task.item()); sum_loss_y += float(loss_y.item()); sum_loss_w += float(loss_w.item()); sum_KL += float(KL_batch.item())
            self._info_trace.append({
                "epoch": epoch, "phase": "warmup" if epoch <= self.warmup_epochs else "dual",
                "objective": "Y_W_joint", "latent_dim": self.latent_dim,
                "loss_task_mean": sum_loss_task / max(n_batches, 1),
                "loss_y_mean": sum_loss_y / max(n_batches, 1),
                "loss_w_mean": sum_loss_w / max(n_batches, 1),
                "KL_mean": sum_KL / max(n_batches, 1), "beta": float(beta), "C": float(C),
            })
        return self

    def predict(self, X):
        import torch
        Xs = self._scaler.transform(X).astype(np.float32)
        X_t = torch.as_tensor(Xs, dtype=torch.float32, device=self._device)
        self._model.eval()
        with torch.no_grad():
            _, _, logits_y, logits_w = self._model(X_t)
            return {
                "y_prob": torch.sigmoid(logits_y).cpu().numpy(),
                "w_pred": torch.sigmoid(logits_w).cpu().numpy(),
            }



# ---------------------------------------------------------------------------
# Deterministic two-tower dot-product latent models
# ---------------------------------------------------------------------------
class _TwoTowerMixin:
    """Two-tower deterministic interaction model.

    The input X is still the same concatenated feature matrix used everywhere
    else in the pipeline: concat([Xb_filtered, Xv_filtered], axis=1).  The model
    only uses xb_dim to split that matrix internally into bacterial and viral
    ProC blocks.

    Pair latent space exported for thesis analysis:
        z_ij = h_b(i) * h_v(j)   (elementwise product)
    Dot-product compatibility used for prediction:
        score_ij = sum_k z_ij,k
    """
    def _init_twotower_params(self, xb_dim=None, latent_dim=5, hidden_dim=64,
                              dropout=0.3, batch_size=512, lr=3e-4,
                              weight_decay=1e-3, n_epochs=200,
                              lambda_y=1.0, lambda_w=1.0,
                              random_state=None, **_):
        if xb_dim is None:
            raise ValueError("Two-tower models require xb_dim in params. The runner injects this automatically.")
        self.xb_dim = int(xb_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.n_epochs = int(n_epochs)
        self.lambda_y = float(lambda_y)
        self.lambda_w = float(lambda_w)
        self.random_state = random_state
        self._scaler = StandardScaler()
        self._device = None
        self._model = None

    def _set_seed(self):
        if self.random_state is not None:
            random.seed(int(self.random_state))
            np.random.seed(int(self.random_state))
            try:
                import torch
                torch.manual_seed(int(self.random_state))
                torch.cuda.manual_seed_all(int(self.random_state))
            except Exception:
                pass

    def _latent_numpy(self, X, batch_size=None):
        import torch
        if batch_size is None:
            batch_size = self.batch_size * 8
        Xs = self._scaler.transform(X).astype(np.float32)
        zs = []
        self._model.eval()
        with torch.no_grad():
            for start in range(0, Xs.shape[0], batch_size):
                xb = torch.as_tensor(Xs[start:start + batch_size], dtype=torch.float32, device=self._device)
                z, _, _, _ = self._model.encode(xb)
                zs.append(z.cpu().numpy())
        return np.vstack(zs)

    def encode_latent(self, X, batch_size=None):
        """Return the pair-level latent interaction vector z = h_b * h_v."""
        return self._latent_numpy(X, batch_size=batch_size)


def _make_twotower_net(x_dim, xb_dim, latent_dim, hidden_dim, dropout, mode):
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            xv_dim = int(x_dim - xb_dim)
            if xv_dim <= 0:
                raise ValueError(f"Invalid two-tower split: x_dim={x_dim}, xb_dim={xb_dim}")
            self.xb_dim = int(xb_dim)
            self.bact_enc = nn.Sequential(
                nn.Linear(self.xb_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim, latent_dim),
            )
            self.virus_enc = nn.Sequential(
                nn.Linear(xv_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim, latent_dim),
            )
            self.bias_y = nn.Parameter(torch.zeros(())) if mode in ("y", "yw") else None
            self.bias_w = nn.Parameter(torch.zeros(())) if mode in ("w", "yw") else None

        def encode(self, X):
            Xb = X[:, :self.xb_dim]
            Xv = X[:, self.xb_dim:]
            hb = self.bact_enc(Xb)
            hv = self.virus_enc(Xv)
            z = hb * hv
            score = z.sum(dim=1)
            return z, hb, hv, score

        def forward(self, X):
            z, hb, hv, score = self.encode(X)
            out = {"z": z, "hb": hb, "hv": hv, "score": score}
            if self.bias_y is not None:
                out["logit_y"] = score + self.bias_y
            if self.bias_w is not None:
                out["pred_w"] = score + self.bias_w
            return out

    return Net()


class TwoTowerYClassifier(BaseClassifier, _TwoTowerMixin):
    name = "twotower_y_only"
    task_type = "binary"

    def __init__(self, **params):
        self._init_twotower_params(**params)

    def fit(self, X, y):
        import torch
        import torch.nn.functional as F
        from torch.utils.data import TensorDataset, DataLoader

        self._set_seed()
        Xs = self._scaler.fit_transform(X).astype(np.float32)
        y = np.asarray(y).astype(np.float32)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_t = torch.as_tensor(Xs, dtype=torch.float32, device=self._device)
        y_t = torch.as_tensor(y, dtype=torch.float32, device=self._device)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True)
        self._model = _make_twotower_net(X_t.shape[1], self.xb_dim, self.latent_dim,
                                         self.hidden_dim, self.dropout, mode="y").to(self._device)
        opt = torch.optim.AdamW(self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        pos_frac = float(y_t.mean().item())
        pos_weight = torch.tensor((1.0 - pos_frac) / pos_frac, dtype=torch.float32, device=self._device) if pos_frac > 0 else None

        for _ in range(self.n_epochs):
            self._model.train()
            for xb, yb in loader:
                opt.zero_grad()
                logits = self._model(xb)["logit_y"]
                loss = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight)
                loss.backward(); opt.step()
        return self

    def predict_proba(self, X):
        import torch
        Xs = self._scaler.transform(X).astype(np.float32)
        X_t = torch.as_tensor(Xs, dtype=torch.float32, device=self._device)
        self._model.eval()
        with torch.no_grad():
            logits = self._model(X_t)["logit_y"]
            return torch.sigmoid(logits).cpu().numpy()


class TwoTowerWRegressor(BaseRegressor, _TwoTowerMixin):
    name = "twotower_w_only"
    task_type = "regression"

    def __init__(self, **params):
        self._init_twotower_params(**params)

    def fit(self, X, y):
        import torch
        import torch.nn.functional as F
        from torch.utils.data import TensorDataset, DataLoader

        self._set_seed()
        Xs = self._scaler.fit_transform(X).astype(np.float32)
        y = np.asarray(y).astype(np.float32)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_t = torch.as_tensor(Xs, dtype=torch.float32, device=self._device)
        y_t = torch.as_tensor(y, dtype=torch.float32, device=self._device)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True)
        self._model = _make_twotower_net(X_t.shape[1], self.xb_dim, self.latent_dim,
                                         self.hidden_dim, self.dropout, mode="w").to(self._device)
        opt = torch.optim.AdamW(self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for _ in range(self.n_epochs):
            self._model.train()
            for xb, yb in loader:
                opt.zero_grad()
                pred = self._model(xb)["pred_w"]
                loss = F.mse_loss(pred, yb)
                loss.backward(); opt.step()
        return self

    def predict(self, X):
        import torch
        Xs = self._scaler.transform(X).astype(np.float32)
        X_t = torch.as_tensor(Xs, dtype=torch.float32, device=self._device)
        self._model.eval()
        with torch.no_grad():
            return self._model(X_t)["pred_w"].cpu().numpy()


class TwoTowerYWJoint(_TwoTowerMixin):
    name = "twotower_yw_joint"
    task_type = "joint_y_w"

    def __init__(self, **params):
        self._init_twotower_params(**params)

    def fit(self, X, y, w):
        import torch
        import torch.nn.functional as F
        from torch.utils.data import TensorDataset, DataLoader

        self._set_seed()
        Xs = self._scaler.fit_transform(X).astype(np.float32)
        y = np.asarray(y).astype(np.float32)
        w = np.asarray(w).astype(np.float32)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_t = torch.as_tensor(Xs, dtype=torch.float32, device=self._device)
        y_t = torch.as_tensor(y, dtype=torch.float32, device=self._device)
        w_t = torch.as_tensor(w, dtype=torch.float32, device=self._device)
        loader = DataLoader(TensorDataset(X_t, y_t, w_t), batch_size=self.batch_size, shuffle=True)
        self._model = _make_twotower_net(X_t.shape[1], self.xb_dim, self.latent_dim,
                                         self.hidden_dim, self.dropout, mode="yw").to(self._device)
        opt = torch.optim.AdamW(self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        pos_frac = float(y_t.mean().item())
        pos_weight = torch.tensor((1.0 - pos_frac) / pos_frac, dtype=torch.float32, device=self._device) if pos_frac > 0 else None
        w_pos_frac = float(w_t.mean().item())
        w_pos_weight = torch.tensor((1.0 - w_pos_frac) / w_pos_frac, dtype=torch.float32, device=self._device) if 0.0 < w_pos_frac < 1.0 else None

        for _ in range(self.n_epochs):
            self._model.train()
            for xb, yb, wb in loader:
                opt.zero_grad()
                out = self._model(xb)
                loss_y = F.binary_cross_entropy_with_logits(out["logit_y"], yb, pos_weight=pos_weight)
                loss_w = F.binary_cross_entropy_with_logits(out["pred_w"], wb, pos_weight=w_pos_weight)
                loss = self.lambda_y * loss_y + self.lambda_w * loss_w
                loss.backward(); opt.step()
        return self

    def predict(self, X):
        import torch
        Xs = self._scaler.transform(X).astype(np.float32)
        X_t = torch.as_tensor(Xs, dtype=torch.float32, device=self._device)
        self._model.eval()
        with torch.no_grad():
            out = self._model(X_t)
            return {
                "y_prob": torch.sigmoid(out["logit_y"]).cpu().numpy(),
                "w_pred": torch.sigmoid(out["pred_w"]).cpu().numpy(),
            }


class XGBRegressorWrap(BaseRegressor):
    name = "xgb_reg"
    task_type = "regression"

    def __init__(self, n_estimators=500, max_depth=4, learning_rate=0.05,
                 subsample=0.8, colsample_bytree=0.8, random_state=None, **_):
        from xgboost import XGBRegressor
        self._XGBRegressor = XGBRegressor
        self.params = dict(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective="reg:squarederror", n_jobs=1, verbosity=0,
            random_state=random_state,
        )
        self._model = None

    def fit(self, X, y):
        self._model = self._XGBRegressor(**self.params)
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Joint MLP baseline: one shared trunk, two heads (Y and W_class)
# ---------------------------------------------------------------------------
def _make_mlp_yw_net(x_dim, hidden_dim, dropout):
    import torch.nn as nn
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin   = nn.Linear(x_dim, hidden_dim)
            self.act   = nn.ReLU()
            self.drop  = nn.Dropout(dropout)
            self.head_y = nn.Linear(hidden_dim, 1)
            self.head_w = nn.Linear(hidden_dim, 1)
        def encode(self, X):
            # post-ReLU hidden activation = the shared joint latent (pre-dropout,
            # matching the separate-MLP encode_latent convention).
            return self.act(self.lin(X))
        def forward(self, X):
            h = self.drop(self.encode(X))
            return self.head_y(h).squeeze(-1), self.head_w(h).squeeze(-1)
    return Net()


class MLPYWJoint:
    """Plain MLP with a shared hidden trunk and two classification heads.

    This is the joint counterpart to the separate mlp_clf / mlp_reg baselines:
    a single Linear->ReLU->Dropout trunk feeds a Y head and a W-class head, both
    trained with BCE. The shared post-ReLU hidden activation is the joint latent
    exported for analysis (encode_latent), directly comparable to the two-tower
    and VIB joint latents.
    """
    name = "mlp_yw_joint"
    task_type = "joint_y_w"

    def __init__(self, hidden_dim=64, dropout=0.3, batch_size=512, lr=3e-4,
                 weight_decay=1e-3, n_epochs=200, lambda_y=1.0, lambda_w=1.0,
                 random_state=None, **_):
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.n_epochs = int(n_epochs)
        self.lambda_y = float(lambda_y)
        self.lambda_w = float(lambda_w)
        self.random_state = random_state
        self._scaler = StandardScaler()
        self._model = None
        self._device = None

    def _set_seed(self):
        if self.random_state is not None:
            random.seed(int(self.random_state))
            np.random.seed(int(self.random_state))
            try:
                import torch
                torch.manual_seed(int(self.random_state))
                torch.cuda.manual_seed_all(int(self.random_state))
            except Exception:
                pass

    def fit(self, X, y, w):
        import torch
        import torch.nn.functional as F
        from torch.utils.data import TensorDataset, DataLoader

        self._set_seed()
        Xs = self._scaler.fit_transform(X).astype(np.float32)
        y = np.asarray(y).astype(np.float32)
        w = np.asarray(w).astype(np.float32)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_t = torch.as_tensor(Xs, dtype=torch.float32, device=self._device)
        y_t = torch.as_tensor(y, dtype=torch.float32, device=self._device)
        w_t = torch.as_tensor(w, dtype=torch.float32, device=self._device)
        loader = DataLoader(TensorDataset(X_t, y_t, w_t),
                            batch_size=self.batch_size, shuffle=True)
        self._model = _make_mlp_yw_net(X_t.shape[1], self.hidden_dim, self.dropout).to(self._device)
        opt = torch.optim.AdamW(self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        def _posw(t):
            f = float(t.mean().item())
            return torch.tensor((1.0 - f) / f, dtype=torch.float32, device=self._device) if 0.0 < f < 1.0 else None
        pw_y, pw_w = _posw(y_t), _posw(w_t)

        for _ in range(self.n_epochs):
            self._model.train()
            for xb, yb, wb in loader:
                opt.zero_grad()
                logit_y, logit_w = self._model(xb)
                loss_y = F.binary_cross_entropy_with_logits(logit_y, yb, pos_weight=pw_y)
                loss_w = F.binary_cross_entropy_with_logits(logit_w, wb, pos_weight=pw_w)
                loss = self.lambda_y * loss_y + self.lambda_w * loss_w
                loss.backward(); opt.step()
        return self

    def predict(self, X):
        import torch
        Xs = self._scaler.transform(X).astype(np.float32)
        X_t = torch.as_tensor(Xs, dtype=torch.float32, device=self._device)
        self._model.eval()
        with torch.no_grad():
            logit_y, logit_w = self._model(X_t)
            return {
                "y_prob": torch.sigmoid(logit_y).cpu().numpy(),
                "w_pred": torch.sigmoid(logit_w).cpu().numpy(),
            }

    def encode_latent(self, X, batch_size=None):
        import torch
        if batch_size is None:
            batch_size = self.batch_size * 8
        Xs = self._scaler.transform(X).astype(np.float32)
        hs = []
        self._model.eval()
        with torch.no_grad():
            for start in range(0, Xs.shape[0], batch_size):
                xb = torch.as_tensor(Xs[start:start + batch_size], dtype=torch.float32, device=self._device)
                hs.append(self._model.encode(xb).cpu().numpy())
        return np.vstack(hs)


MODELS = {
    "sparse_logistic": SparseLogistic,
    "xgb_clf":         XGBClassifierWrap,
    "mlp_clf":         MLPClassifier,
    "linreg":          LinearRegressor,
    "xgb_reg":         XGBRegressorWrap,
    "mlp_reg":         MLPRegressor,
    "vib_y_only":      VIBYClassifier,
    "vib_w_only":      VIBWRegressor,
    "vib_yw_joint":    VIBYWJoint,
    "twotower_y_only":      TwoTowerYClassifier,
    "twotower_w_only":      TwoTowerWRegressor,
    "twotower_yw_joint":    TwoTowerYWJoint,
    "mlp_yw_joint":         MLPYWJoint,
}


def build_model(class_key: str, **params):
    if class_key not in MODELS:
        raise KeyError(f"Unknown model '{class_key}'. Available: {list(MODELS)}")
    return MODELS[class_key](**params)
