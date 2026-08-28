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

# Base interfaces
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

# Classifiers
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

# Regressors
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

# Registry
# Joint MLP baseline: one shared trunk, two heads (Y and W_class)
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
    "mlp_yw_joint":    MLPYWJoint,
}


def build_model(class_key: str, **params):
    if class_key not in MODELS:
        raise KeyError(f"Unknown model '{class_key}'. Available: {list(MODELS)}")
    return MODELS[class_key](**params)
