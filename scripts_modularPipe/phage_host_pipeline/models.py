"""Model registry. Binary models expose fit/predict_proba, regressors fit/predict.

Adding a model = one class here + one entry in MODELS + one entry in config.yaml.
"""
from __future__ import annotations

import random
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning, message=".*penalty.*was deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Inconsistent values: penalty.*")


# =============================================================================
# Linear models
# =============================================================================
class SparseLogistic:
    """L1 logistic regression (standardised, balanced class weights).

    With `lambda_cv.enabled`, the penalty is chosen by K-fold CV inside the
    training data over a grid of lambda / lambda_max ratios, then refit.
    """

    def __init__(self, C=1.0, penalty="l1", solver="liblinear", max_iter=2000,
                 random_state=None, lambda_cv=None, **_):
        self.C, self.penalty, self.solver, self.max_iter = C, penalty, solver, max_iter
        self.random_state = random_state
        self.lambda_cv = lambda_cv if lambda_cv and lambda_cv.get("enabled", True) else None
        self.cv_record, self.cv_curve = None, []

    def _logreg(self, C):
        return LogisticRegression(C=C, penalty=self.penalty, solver=self.solver,
                                  max_iter=self.max_iter, class_weight="balanced",
                                  random_state=self.random_state)

    @staticmethod
    def _lambda_max(Xs, y):
        return float(np.abs(Xs.T @ (np.asarray(y, float) - np.mean(y))).max() / len(y))

    def _choose_C(self, X, y):
        cfg = self.lambda_cv
        k, n_points = int(cfg.get("n_inner", 3)), int(cfg.get("n_points", 8))
        metric, rule = str(cfg.get("metric", "auc")).lower(), str(cfg.get("rule", "1se")).lower()
        y = np.asarray(y).astype(int)
        if len(np.unique(y)) < 2 or np.bincount(y).min() < k:
            return self.C                                   # too small to tune

        ratios = np.exp(np.linspace(0.0, np.log(float(cfg.get("ratio_min", 0.05))), n_points))
        seed = 0 if self.random_state is None else int(self.random_state)
        folds = list(StratifiedKFold(k, shuffle=True, random_state=seed).split(X, y))
        scores = np.full((n_points, k), np.nan)
        for i, r in enumerate(ratios):
            for j, (tr, va) in enumerate(folds):
                sc = StandardScaler().fit(X[tr])
                A, B = sc.transform(X[tr]), sc.transform(X[va])
                lmax = self._lambda_max(A, y[tr])
                if not np.isfinite(lmax) or lmax <= 0:
                    continue
                pv = self._logreg(1.0 / (len(tr) * lmax * r)).fit(A, y[tr]).predict_proba(B)[:, 1]
                if metric == "auc":
                    if len(np.unique(y[va])) > 1:
                        scores[i, j] = roc_auc_score(y[va], pv)
                else:                                       # deviance, negated
                    scores[i, j] = -2.0 * log_loss(y[va], np.clip(pv, 1e-9, 1 - 1e-9))

        mu = np.nanmean(scores, axis=1)
        if not np.isfinite(mu).any():
            return self.C
        se = np.nanstd(scores, axis=1, ddof=1) / np.sqrt(np.sum(~np.isnan(scores), axis=1))
        best = int(np.nanargmax(mu))
        if rule == "1se":   # largest lambda within one SE of the best
            pick = int(np.flatnonzero(mu >= mu[best] - (se[best] if np.isfinite(se[best]) else 0.0)).min())
        else:
            pick = best

        lmax_full = self._lambda_max(StandardScaler().fit_transform(X), y)
        chosen_C = 1.0 / (len(y) * lmax_full * ratios[pick])
        rnd = lambda v, d: round(float(v), d) if np.isfinite(v) else None
        self.cv_curve = [{"metric": metric, "ratio": round(float(r), 6), "score_mean": rnd(m, 5),
                          "score_se": rnd(s, 5), "is_min": i == best, "is_1se": i == pick}
                         for i, (r, m, s) in enumerate(zip(ratios, mu, se))]
        self.cv_record = {"metric": metric, "rule": rule, "n_inner": k, "n_points": n_points,
                          "ratio_min": float(cfg.get("ratio_min", 0.05)),
                          "lambda_max": round(lmax_full, 8),
                          "chosen_ratio": round(float(ratios[pick]), 6), "chosen_C": float(chosen_C),
                          "chosen_score": round(float(mu[pick]), 5),
                          "best_ratio": round(float(ratios[best]), 6), "best_score": round(float(mu[best]), 5),
                          "best_score_se": rnd(se[best], 5)}
        return chosen_C

    def fit(self, X, y):
        C = self._choose_C(np.asarray(X, dtype=float), y) if self.lambda_cv else self.C
        self._scaler = StandardScaler()
        self._model = self._logreg(C).fit(self._scaler.fit_transform(X), y)
        if self.cv_record:
            self.cv_record["n_nonzero"] = int((np.abs(self._model.coef_[0]) > 0).sum())
        return self

    def predict_proba(self, X):
        return self._model.predict_proba(self._scaler.transform(X))[:, 1]


class LinearRegressor:
    """Standardised ridge regression."""

    def __init__(self, alpha=1.0, random_state=None, **_):
        self.alpha, self.random_state = alpha, random_state

    def fit(self, X, y):
        self._scaler = StandardScaler()
        self._model = Ridge(alpha=self.alpha, random_state=self.random_state)
        self._model.fit(self._scaler.fit_transform(X), y)
        return self

    def predict(self, X):
        return self._model.predict(self._scaler.transform(X))


# =============================================================================
# XGBoost
# =============================================================================
XGB_DEFAULTS = dict(n_estimators=500, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8)


class XGBClassifierWrap:
    def __init__(self, random_state=None, **params):
        self.params = {k: params.get(k, v) for k, v in XGB_DEFAULTS.items()}
        self.random_state = random_state

    def fit(self, X, y):
        from xgboost import XGBClassifier
        n_pos, n_neg = int(np.sum(y == 1)), int(np.sum(y == 0))
        self._model = XGBClassifier(scale_pos_weight=n_neg / n_pos if n_pos else 1.0,
                                    eval_metric="logloss", n_jobs=4, verbosity=1,
                                    random_state=self.random_state, **self.params)
        self._model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self._model.predict_proba(X)[:, 1]


class XGBRegressorWrap:
    def __init__(self, random_state=None, **params):
        self.params = {k: params.get(k, v) for k, v in XGB_DEFAULTS.items()}
        self.random_state = random_state

    def fit(self, X, y):
        from xgboost import XGBRegressor
        self._model = XGBRegressor(objective="reg:squarederror", n_jobs=1, verbosity=0,
                                   random_state=self.random_state, **self.params)
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)


# =============================================================================
# MLPs: Linear -> ReLU -> Dropout -> one Linear head per target
# =============================================================================
class TorchMLP:
    """One-hidden-layer MLP. `losses` gives one head per target ("bce" or "mse").

    The post-ReLU hidden activation is exposed as the latent (encode_latent).
    """
    losses: tuple = ()

    def __init__(self, hidden_dim=64, dropout=0.3, batch_size=512, lr=3e-4,
                 weight_decay=1e-3, n_epochs=200, random_state=None, weights=None, **_):
        self.hidden_dim, self.dropout = int(hidden_dim), float(dropout)
        self.batch_size, self.lr, self.weight_decay = int(batch_size), float(lr), float(weight_decay)
        self.n_epochs, self.random_state = int(n_epochs), random_state
        self.weights = weights or [1.0] * len(self.losses)

    def _net(self, x_dim):
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.lin = nn.Linear(x_dim, self.hidden_dim)
                s.act = nn.ReLU()
                s.drop = nn.Dropout(self.dropout)
                s.heads = nn.ModuleList([nn.Linear(self.hidden_dim, 1) for _ in self.losses])

            def encode(s, x):
                return s.act(s.lin(x))

            def forward(s, x):
                h = s.drop(s.encode(x))
                return [head(h).squeeze(-1) for head in s.heads]
        return Net()

    def _tensor(self, X):
        import torch
        return torch.as_tensor(self._scaler.transform(X).astype(np.float32),
                               dtype=torch.float32, device=self._device)

    def fit(self, X, *targets):
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset

        if self.random_state is not None:
            random.seed(int(self.random_state))
            np.random.seed(int(self.random_state))
            torch.manual_seed(int(self.random_state))
            torch.cuda.manual_seed_all(int(self.random_state))
        # Deterministic kernels (no effect on CPU runs; removes cuDNN nondeterminism on GPU).
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self._scaler = StandardScaler()
        Xs = self._scaler.fit_transform(X).astype(np.float32)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_t = torch.as_tensor(Xs, dtype=torch.float32, device=self._device)
        T = [torch.as_tensor(np.asarray(t).astype(np.float32), dtype=torch.float32,
                             device=self._device) for t in targets]
        loader = DataLoader(TensorDataset(X_t, *T), batch_size=self.batch_size, shuffle=True)
        self._model = self._net(X_t.shape[1]).to(self._device)
        opt = torch.optim.AdamW(self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        pos_w = []                      # BCE weight = n_neg / n_pos
        for kind, t in zip(self.losses, T):
            f = float(t.mean().item())
            pos_w.append(torch.tensor((1.0 - f) / f, dtype=torch.float32, device=self._device)
                         if kind == "bce" and 0.0 < f < 1.0 else None)

        for _ in range(self.n_epochs):
            self._model.train()
            for xb, *tb in loader:
                opt.zero_grad()
                losses = [F.binary_cross_entropy_with_logits(o, t, pos_weight=pw) if kind == "bce"
                          else F.mse_loss(o, t)
                          for kind, o, t, pw in zip(self.losses, self._model(xb), tb, pos_w)]
                loss = losses[0] if len(losses) == 1 else self.weights[0] * losses[0]
                for w, l in zip(self.weights[1:], losses[1:]):
                    loss = loss + w * l
                loss.backward()
                opt.step()
        return self

    def _outputs(self, X, sigmoid=False):
        import torch
        self._model.eval()
        with torch.no_grad():
            out = self._model(self._tensor(X))
            return [(torch.sigmoid(o) if sigmoid else o).cpu().numpy() for o in out]

    def encode_latent(self, X, batch_size=None):
        import torch
        bs = batch_size or self.batch_size * 8
        self._model.eval()
        with torch.no_grad():
            return np.vstack([self._model.encode(self._tensor(X[i:i + bs])).cpu().numpy()
                              for i in range(0, X.shape[0], bs)])


class MLPClassifier(TorchMLP):
    losses = ("bce",)

    def predict_proba(self, X):
        return self._outputs(X, sigmoid=True)[0]


class MLPRegressor(TorchMLP):
    losses = ("mse",)

    def predict(self, X):
        return self._outputs(X)[0]


class MLPYWJoint(MLPClassifier):
    """Shared trunk, one BCE head for y and one for w_class."""
    losses = ("bce", "bce")

    def __init__(self, lambda_y=1.0, lambda_w=1.0, **kw):
        super().__init__(weights=[float(lambda_y), float(lambda_w)], **kw)

    def predict(self, X):
        y_prob, w_prob = self._outputs(X, sigmoid=True)
        return {"y_prob": y_prob, "w_pred": w_prob}


MODELS = {
    "sparse_logistic": SparseLogistic,
    "xgb_clf": XGBClassifierWrap,
    "mlp_clf": MLPClassifier,
    "linreg": LinearRegressor,
    "xgb_reg": XGBRegressorWrap,
    "mlp_reg": MLPRegressor,
    "mlp_yw_joint": MLPYWJoint,
}


def build_model(class_key: str, **params):
    if class_key not in MODELS:
        raise KeyError(f"Unknown model '{class_key}'. Available: {list(MODELS)}")
    return MODELS[class_key](**params)
