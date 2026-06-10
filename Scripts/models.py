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
MODELS = {
    "sparse_logistic": SparseLogistic,
    "xgb_clf":         XGBClassifierWrap,
    "linreg":          LinearRegressor,
    "xgb_reg":         XGBRegressorWrap,
}


def build_model(class_key: str, **params):
    if class_key not in MODELS:
        raise KeyError(f"Unknown model '{class_key}'. Available: {list(MODELS)}")
    return MODELS[class_key](**params)
