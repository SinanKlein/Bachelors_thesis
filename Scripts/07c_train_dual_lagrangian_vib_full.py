# 07c_train_dual_lagrangian_vib_full_STRICT_SINGLE_CHANNEL_lat1to6.py
#   Bottleneck model, raw X (no CLR), multiple train/val/test splits
#   Non-linear encoder and non-linear heads on latent z
#   Stochastic latent space
#   Tasks:
#     - Y | X         (binary)
#     - W>0 | X       (binary)
#     - W | X         (continuous regression)
#     - W_sign+ | X   (binary: 1 if W_sign = +1)
#     - W_sign- | X   (binary: 1 if W_sign = -1)
#
#   Latent dimension is configurable via config["latent_dim"], default = 1.
#   Option: config["use_W_tasks"] toggles whether W and W>0 contribute to the loss.


import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, r2_score
from scipy.stats import spearmanr
import random

from sklearn.linear_model import LogisticRegression, LinearRegression


# ------------------------------------------------------------------
# Paths (Windows) 
# ------------------------------------------------------------------
PROJECT_ROOT = r"C:\Sinan_Klein\LMU\lmu_thesis\multimodal_network"
DATA_DIR = r"C:\Sinan_Klein\LMU\lmu_thesis\multimodal_network\data\data_k-20260127T100440Z-3-001\data_k"

MODEL_DIR  = os.path.join(PROJECT_ROOT, "models_vib_dual")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs_vib_dual")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

SPLIT_TEMPLATE     = os.path.join(DATA_DIR, "train_val_test_split_k{}.npz")
MODEL_TEMPLATE_NN  = os.path.join(MODEL_DIR, "bottleneck_baseline_lat{d}_k{k}.pt")
MODEL_TEMPLATE_VIB = os.path.join(MODEL_DIR, "bottleneck_vib_strict_single_lat{d}_k{k}.pt")

# ------------------------------------------------------------------
# 0. Config 
# ------------------------------------------------------------------
config = {
    # Original baseline parameters
    "latent_dim": 3,    # doesnt really matter because a loop that tires dimensions 1:10 is created
    "hidden_dim": 64,
    "dropout": 0.3,
    "head_hidden_dim": 32,  
    "lambda_y": 1.0,
    "lambda_w": 1.0,
    "lambda_s": 2.0,
    "use_W_tasks": True,
    "batch_size": 512,
    "lr": 3e-4,
    "weight_decay": 1e-3,
    "n_epochs": 200,
    "seed": 0,
    "patience": 20,
    "n_splits": 10,

    # VIB parameters 
    "warmup_epochs": 60,
    "dual_lr": 1e-3,
    "beta_init": 1e-4,
    "beta_max": 10.0,
    "m": 0.8,               # used as the single budget multiplier m (strict single)
    "C_min": 0.05,
    "use_moving_avg": True,
    "KL_smoothing": 0.99
}


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------
# 1. ORIGINAL Bottleneck model (Baseline) (same logic as your code)
# ------------------------------------------------------------------
class BottleneckMultiHeadNonlinear(nn.Module):
    def __init__(self, x_dim, latent_dim=1, hidden_dim=64, dropout=0.3, head_hidden_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim)
        )

        def make_head():
            return nn.Sequential(nn.Linear(latent_dim, 1))

        self.head_Y    = make_head()
        self.head_W0   = make_head()
        self.head_W    = make_head()
        self.head_Spos = make_head()
        self.head_Sneg = make_head()

    def forward(self, X):
        z = self.encoder(X)
        logit_y    = self.head_Y(z).squeeze(-1)
        logit_w0   = self.head_W0(z).squeeze(-1)
        pred_w     = self.head_W(z).squeeze(-1)
        logit_spos = self.head_Spos(z).squeeze(-1)
        logit_sneg = self.head_Sneg(z).squeeze(-1)
        return z, logit_y, logit_w0, pred_w, logit_spos, logit_sneg

    @torch.no_grad()
    def encode(self, X):
        return self.encoder(X)


def make_loss_fn(pos_weight_y, lambda_y=1.0, lambda_w=1.0, lambda_s=1.0, use_W_tasks=True):
    def loss_fn(model, X, Y, W, S):
        z, logit_y, logit_w0, pred_w, logit_spos, logit_sneg = model(X)
        loss_y = F.binary_cross_entropy_with_logits(logit_y, Y.float(), pos_weight=pos_weight_y)

        if use_W_tasks:
            is_pos = (W > 0).float()
            loss_w0 = F.binary_cross_entropy_with_logits(logit_w0, is_pos)
            loss_w_reg = F.mse_loss(pred_w, W.float())
            loss_w = 0.5 * (loss_w0 + loss_w_reg)
        else:
            loss_w0 = torch.tensor(0.0, device=X.device)
            loss_w_reg = torch.tensor(0.0, device=X.device)
            loss_w = torch.tensor(0.0, device=X.device)

        S_pos, S_neg = (S == 1).float(), (S == -1).float()
        loss_s_pos = F.binary_cross_entropy_with_logits(logit_spos, S_pos)
        loss_s_neg = F.binary_cross_entropy_with_logits(logit_sneg, S_neg)
        loss_s = 0.5 * (loss_s_pos + loss_s_neg)

        loss = lambda_y * loss_y + lambda_w * loss_w + lambda_s * loss_s
        return loss, (
            loss_y.item(),
            loss_w0.item() if use_W_tasks else 0.0,
            loss_w_reg.item() if use_W_tasks else 0.0,
            loss_s_pos.item(),
            loss_s_neg.item()
        )
    return loss_fn


# ------------------------------------------------------------------
# 2. STRICT SINGLE-CHANNEL VIB model
#    - one latent z of dimension latent_dim
#    - all heads use the same z (full sharing)
#    - one KL, one beta, one budget
# ------------------------------------------------------------------
class BottleneckVIBStrictSingle(nn.Module):
    def __init__(self, x_dim, latent_dim=3, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder_base = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # One stochastic channel: outputs mu and logvar for the whole latent
        self.enc_Z = nn.Linear(hidden_dim, latent_dim * 2)

        def make_head():
            return nn.Sequential(nn.Linear(latent_dim, 1))

        # ALL heads consume the SAME z (full sharing)
        self.head_Y    = make_head()
        self.head_Spos = make_head()
        self.head_Sneg = make_head()
        self.head_W0   = make_head()
        self.head_W    = make_head()

    def encode(self, X):
        h = self.encoder_base(X)
        mu, logvar = torch.chunk(self.enc_Z(h), 2, dim=-1)
        # same guardrails style
        logvar = torch.clamp(logvar, min=-10.0, max=5.0)
        return mu, logvar

    def forward(self, X):
        # deterministic eval: use mu
        mu, logvar = self.encode(X)
        logit_y    = self.head_Y(mu).squeeze(-1)
        logit_spos = self.head_Spos(mu).squeeze(-1)
        logit_sneg = self.head_Sneg(mu).squeeze(-1)
        logit_w0   = self.head_W0(mu).squeeze(-1)
        pred_w     = self.head_W(mu).squeeze(-1)
        return mu, logit_y, logit_w0, pred_w, logit_spos, logit_sneg


def kl_diag_gaussian(mu, logvar):
    # mean over batch
    return 0.5 * torch.mean(torch.sum(torch.exp(logvar) + mu.pow(2) - 1.0 - logvar, dim=1))


def sample_z(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + std * eps


# ------------------------------------------------------------------
# Helper: run one split for one latent_dim
# ------------------------------------------------------------------
def run_one_split(split_idx: int, latent_dim: int):
    print(f"\n================ Split {split_idx} | latent_dim={latent_dim} ================")

    split_path = SPLIT_TEMPLATE.format(split_idx)
    data = np.load(split_path)

    X_train, Y_train, W_train, S_train = data["X_train"], data["Y_train"], data["W_train"], data["W_sign_train"]
    X_val, Y_val, W_val, S_val         = data["X_val"], data["Y_val"], data["W_val"], data["W_sign_val"]
    X_test, Y_test, W_test, S_test     = data["X_test"], data["Y_test"], data["W_test"], data["W_sign_test"]

    set_seed(config["seed"] + split_idx)

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled   = scaler_X.transform(X_val)
    X_test_scaled  = scaler_X.transform(X_test)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def to_torch(x, dtype): return torch.as_tensor(x, dtype=dtype, device=device)

    X_train_t = to_torch(X_train_scaled, torch.float32)
    Y_train_t = to_torch(Y_train, torch.float32)
    W_train_t = to_torch(W_train, torch.float32)
    S_train_t = to_torch(S_train, torch.long)

    X_val_t = to_torch(X_val_scaled, torch.float32)
    Y_val_t = to_torch(Y_val, torch.float32)
    W_val_t = to_torch(W_val, torch.float32)
    S_val_t = to_torch(S_val, torch.long)

    X_test_t = to_torch(X_test_scaled, torch.float32)
    Y_test_t = to_torch(Y_test, torch.float32)
    W_test_t = to_torch(W_test, torch.float32)
    S_test_t = to_torch(S_test, torch.long)

    train_ds = TensorDataset(X_train_t, Y_train_t, W_train_t, S_train_t)
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    x_dim = X_train_t.shape[1]

    pos_frac = float(Y_train_t.mean())
    pos_weight_y = torch.tensor((1.0 - pos_frac) / pos_frac, dtype=torch.float32, device=device)

    # ------------------------------------------------------------------
    # TRAINING MODEL 1: ORIGINAL BASELINE NN
    # ------------------------------------------------------------------
    print("\n--- Training Baseline NN ---")
    model_nn = BottleneckMultiHeadNonlinear(
        x_dim,
        latent_dim=latent_dim,
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
        head_hidden_dim=config["head_hidden_dim"],
    ).to(device)

    opt_nn = torch.optim.AdamW(
        model_nn.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )

    loss_fn_nn = make_loss_fn(
        pos_weight_y,
        lambda_y=config["lambda_y"],
        lambda_w=config["lambda_w"],
        lambda_s=config["lambda_s"],
        use_W_tasks=config["use_W_tasks"]
    )

    best_val_auc_nn, best_state_nn, epochs_no_improve = -np.inf, None, 0

    for epoch in range(1, config["n_epochs"] + 1):
        model_nn.train()
        for Xb, Yb, Wb, Sb in train_loader:
            opt_nn.zero_grad()
            loss, _ = loss_fn_nn(model_nn, Xb, Yb, Wb, Sb)
            loss.backward()
            opt_nn.step()

        model_nn.eval()
        with torch.no_grad():
            _, logit_y_val, _, _, _, _ = model_nn(X_val_t)
            val_auc_y = roc_auc_score(Y_val_t.cpu(), torch.sigmoid(logit_y_val).cpu())

        if val_auc_y > best_val_auc_nn:
            best_val_auc_nn, best_state_nn, epochs_no_improve = val_auc_y, model_nn.state_dict(), 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= config["patience"]:
            break

    if best_state_nn is not None:
        model_nn.load_state_dict(best_state_nn)

    # Save baseline NN model
    torch.save(model_nn.state_dict(), MODEL_TEMPLATE_NN.format(d=latent_dim, k=split_idx))

    model_nn.eval()
    with torch.no_grad():
        _, logit_y_test, logit_w0_test, pred_w_test, logit_spos_test, logit_sneg_test = model_nn(X_test_t)

        auc_y_nn = roc_auc_score(Y_test_t.cpu(), torch.sigmoid(logit_y_test).cpu())
        rmse_W_nn = np.sqrt(np.mean((W_test_t.cpu().numpy() - pred_w_test.cpu().numpy()) ** 2))
        r2_W_nn   = r2_score(W_test_t.cpu(), pred_w_test.cpu())
        rho_W_nn, _ = spearmanr(W_test_t.cpu(), pred_w_test.cpu())

        try:
            auc_w0_nn = roc_auc_score((W_test_t.cpu() > 0).numpy(), torch.sigmoid(logit_w0_test).cpu())
        except:
            auc_w0_nn = np.nan

        try:
            auc_Spos_nn = roc_auc_score((S_test_t.cpu() == 1).numpy(), torch.sigmoid(logit_spos_test).cpu())
        except:
            auc_Spos_nn = np.nan

        try:
            auc_Sneg_nn = roc_auc_score((S_test_t.cpu() == -1).numpy(), torch.sigmoid(logit_sneg_test).cpu())
        except:
            auc_Sneg_nn = np.nan
    # ------------------------------------------------------------------
    # TRAINING MODEL 2: STRICT SINGLE-CHANNEL VIB (one KL, one beta, one budget)
    # ------------------------------------------------------------------
    print("\n--- Training VIB Strict Single-Channel (Dual Ascent) ---")

    model_vib = BottleneckVIBStrictSingle(
        x_dim,
        latent_dim=latent_dim,
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
    ).to(device)

    opt_vib = torch.optim.Adam(model_vib.parameters(), lr=config["lr"])

    beta = config["beta_init"]
    store_KL = []
    C = config["C_min"]          # fallback
    KL_ma = 0.0

    best_val_auc_vib, best_state_vib, patience_vib = -np.inf, None, 0

    # info-theory logging rows (per-epoch)
    info_rows = []

    for epoch in range(1, config["n_epochs"] + 1):
        model_vib.train()

        # Accumulators for epoch-mean logs
        n_batches = 0
        sum_loss_task = 0.0
        sum_loss_y = 0.0
        sum_loss_s = 0.0
        sum_loss_w = 0.0
        sum_KL = 0.0

        for Xb, Yb, Wb, Sb in train_loader:
            opt_vib.zero_grad()

            mu, logvar = model_vib.encode(Xb)
            z = sample_z(mu, logvar)

            yhat = model_vib.head_Y(z).squeeze(-1)
            spos = model_vib.head_Spos(z).squeeze(-1)
            sneg = model_vib.head_Sneg(z).squeeze(-1)
            what = model_vib.head_W(z).squeeze(-1)
            w0hat = model_vib.head_W0(z).squeeze(-1)

            # Losses (same structure as your VIB code)
            loss_y = F.binary_cross_entropy_with_logits(yhat, Yb.float(), pos_weight=pos_weight_y)

            loss_s = 0.5 * (
                F.binary_cross_entropy_with_logits(spos, (Sb == 1).float()) +
                F.binary_cross_entropy_with_logits(sneg, (Sb == -1).float())
            )

            if config["use_W_tasks"]:
                loss_w = 0.5 * (
                    F.binary_cross_entropy_with_logits(w0hat, (Wb > 0).float()) +
                    F.mse_loss(what, Wb.float())
                )
            else:
                loss_w = torch.tensor(0.0, device=device)

            loss_task = config["lambda_y"] * loss_y + config["lambda_s"] * loss_s + config["lambda_w"] * loss_w

            KL_batch = kl_diag_gaussian(mu, logvar) 

            # PHASE 1: WARMUP
            if epoch <= config["warmup_epochs"]:
                loss = loss_task + beta * KL_batch
                loss.backward()
                opt_vib.step()

                if epoch > config["warmup_epochs"] - 10:
                    store_KL.append(KL_batch.item())

            # PHASE 2: CONSTRAINED DUAL ASCENT
            else:
                if epoch == config["warmup_epochs"] + 1 and C == config["C_min"]:
                    KL_med = np.median(store_KL) if store_KL else 1.0
                    C = max(config["C_min"], config["m"] * KL_med)
                    KL_ma = KL_med
                    print(f"  [VIB Budget Configured] C: {C:.3f} (per-dim KL)")

                if config["use_moving_avg"]:
                    KL_ma = config["KL_smoothing"] * KL_ma + (1 - config["KL_smoothing"]) * KL_batch.item()
                    KL_used = KL_ma
                else:
                    KL_used = KL_batch.item()

                loss = loss_task + beta * KL_batch
                loss.backward()
                opt_vib.step()

                # Dual update (single)
                beta = min(config["beta_max"], max(0.0, beta + config["dual_lr"] * (KL_used - C)))

            # Accumulate
            n_batches += 1
            sum_loss_task += float(loss_task.item())
            sum_loss_y += float(loss_y.item())
            sum_loss_s += float(loss_s.item())
            sum_loss_w += float(loss_w.item()) if config["use_W_tasks"] else 0.0
            sum_KL += float(KL_batch.item())

        # ---- End epoch: validation for early stopping (same logic as your VIB code)
        val_auc_vib = np.nan
        if epoch > config["warmup_epochs"]:
            model_vib.eval()
            with torch.no_grad():
                _, logit_y_val, _, _, _, _ = model_vib(X_val_t)
                val_auc_vib = roc_auc_score(Y_val_t.cpu(), torch.sigmoid(logit_y_val).cpu())

            if val_auc_vib > best_val_auc_vib:
                best_val_auc_vib = val_auc_vib
                best_state_vib = model_vib.state_dict()
                patience_vib = 0
            else:
                patience_vib += 1

            if patience_vib >= config["patience"]:
                print(f"  VIB Early stopping at epoch {epoch}. Best Val AUC: {best_val_auc_vib:.3f}")
                # log this epoch too, then break
                pass

        # ---- log epoch (info-theoretic trace)
        if n_batches > 0:
            info_rows.append({
                "latent_dim": latent_dim,
                "split_idx": split_idx,
                "epoch": epoch,
                "phase": "warmup" if epoch <= config["warmup_epochs"] else "dual",
                "loss_task_mean": sum_loss_task / n_batches,
                "loss_y_mean": sum_loss_y / n_batches,
                "loss_s_mean": sum_loss_s / n_batches,
                "loss_w_mean": sum_loss_w / n_batches if config["use_W_tasks"] else 0.0,
                "KL_per_dim_mean": sum_KL / n_batches,
                "beta": float(beta),
                "C": float(C),
                "val_auc_y": float(val_auc_vib) if not np.isnan(val_auc_vib) else np.nan,
            })

        if epoch > config["warmup_epochs"] and patience_vib >= config["patience"]:
            break

    if best_state_vib is not None:
        model_vib.load_state_dict(best_state_vib)

    # Save VIB model
    torch.save(model_vib.state_dict(), MODEL_TEMPLATE_VIB.format(d=latent_dim, k=split_idx))

    # ------------------------------------------------------------------
    # EXTRA METRICS: train/test eval loss, generalization gap,
    # KL_total (NOT per-dim), and n_train
    # ------------------------------------------------------------------
    model_vib.eval()
    with torch.no_grad():
        # Train eval (deterministic z = mu)
        eval_train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=False)
        _sum_loss_task_train = 0.0
        _sum_KL_total_train = 0.0
        _n_train_eval = 0

        for Xb, Yb, Wb, Sb in eval_train_loader:
            mu, logvar = model_vib.encode(Xb)
            z = mu

            yhat = model_vib.head_Y(z).squeeze(-1)
            spos = model_vib.head_Spos(z).squeeze(-1)
            sneg = model_vib.head_Sneg(z).squeeze(-1)
            what = model_vib.head_W(z).squeeze(-1)
            w0hat = model_vib.head_W0(z).squeeze(-1)

            loss_y_eval = F.binary_cross_entropy_with_logits(yhat, Yb.float(), pos_weight=pos_weight_y)

            loss_s_eval = 0.5 * (
                F.binary_cross_entropy_with_logits(spos, (Sb == 1).float()) +
                F.binary_cross_entropy_with_logits(sneg, (Sb == -1).float())
            )

            if config["use_W_tasks"]:
                loss_w_eval = 0.5 * (
                    F.binary_cross_entropy_with_logits(w0hat, (Wb > 0).float()) +
                    F.mse_loss(what, Wb.float())
                )
            else:
                loss_w_eval = torch.tensor(0.0, device=device)

            loss_task_eval = config["lambda_y"] * loss_y_eval + config["lambda_s"] * loss_s_eval + config["lambda_w"] * loss_w_eval
            KL_total_batch = kl_diag_gaussian(mu, logvar)

            bs = Xb.shape[0]
            _sum_loss_task_train += float(loss_task_eval.item()) * bs
            _sum_KL_total_train += float(KL_total_batch.item()) * bs
            _n_train_eval += int(bs)

        train_loss_eval_vib = _sum_loss_task_train / _n_train_eval if _n_train_eval > 0 else np.nan
        KL_total_vib = _sum_KL_total_train / _n_train_eval if _n_train_eval > 0 else np.nan

        # Test eval (deterministic z = mu)
        test_ds = TensorDataset(X_test_t, Y_test_t, W_test_t, S_test_t)
        eval_test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False)
        _sum_loss_task_test = 0.0
        _n_test_eval = 0

        for Xb, Yb, Wb, Sb in eval_test_loader:
            mu, logvar = model_vib.encode(Xb)
            z = mu

            yhat = model_vib.head_Y(z).squeeze(-1)
            spos = model_vib.head_Spos(z).squeeze(-1)
            sneg = model_vib.head_Sneg(z).squeeze(-1)
            what = model_vib.head_W(z).squeeze(-1)
            w0hat = model_vib.head_W0(z).squeeze(-1)

            loss_y_eval = F.binary_cross_entropy_with_logits(yhat, Yb.float(), pos_weight=pos_weight_y)

            loss_s_eval = 0.5 * (
                F.binary_cross_entropy_with_logits(spos, (Sb == 1).float()) +
                F.binary_cross_entropy_with_logits(sneg, (Sb == -1).float())
            )

            if config["use_W_tasks"]:
                loss_w_eval = 0.5 * (
                    F.binary_cross_entropy_with_logits(w0hat, (Wb > 0).float()) +
                    F.mse_loss(what, Wb.float())
                )
            else:
                loss_w_eval = torch.tensor(0.0, device=device)

            loss_task_eval = config["lambda_y"] * loss_y_eval + config["lambda_s"] * loss_s_eval + config["lambda_w"] * loss_w_eval

            bs = Xb.shape[0]
            _sum_loss_task_test += float(loss_task_eval.item()) * bs
            _n_test_eval += int(bs)

        test_loss_eval_vib = _sum_loss_task_test / _n_test_eval if _n_test_eval > 0 else np.nan
        gen_gap_vib = test_loss_eval_vib - train_loss_eval_vib
        n_train = int(X_train.shape[0])

    # Test metrics for VIB
    model_vib.eval()
    with torch.no_grad():
        _, logit_y_test, logit_w0_test, pred_w_test, logit_spos_test, logit_sneg_test = model_vib(X_test_t)

        auc_y_vib = roc_auc_score(Y_test_t.cpu(), torch.sigmoid(logit_y_test).cpu())
        rmse_W_vib = np.sqrt(np.mean((W_test_t.cpu().numpy() - pred_w_test.cpu().numpy()) ** 2))
        r2_W_vib   = r2_score(W_test_t.cpu(), pred_w_test.cpu())
        rho_W_vib, _ = spearmanr(W_test_t.cpu(), pred_w_test.cpu())

        try:
            auc_w0_vib = roc_auc_score((W_test_t.cpu() > 0).numpy(), torch.sigmoid(logit_w0_test).cpu())
        except:
            auc_w0_vib = np.nan

        try:
            auc_Spos_vib = roc_auc_score((S_test_t.cpu() == 1).numpy(), torch.sigmoid(logit_spos_test).cpu())
        except:
            auc_Spos_vib = np.nan

        try:
            auc_Sneg_vib = roc_auc_score((S_test_t.cpu() == -1).numpy(), torch.sigmoid(logit_sneg_test).cpu())
        except:
            auc_Sneg_vib = np.nan

    print(f"  VIB Test Y: AUC={auc_y_vib:.3f} | W: R²={r2_W_vib:.3f} | W>0: AUC={auc_w0_vib:.3f}")

    # ------------------------------------------------------------------
    # ORIGINAL BASELINES (Sklearn) (same logic)
    # ------------------------------------------------------------------
    X_train_np, X_test_np = X_train_scaled, X_test_scaled

    logreg_Y = LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, class_weight="balanced"
    ).fit(X_train_np, Y_train)
    auc_y_lr = roc_auc_score(Y_test, logreg_Y.predict_proba(X_test_np)[:, 1])

    is_pos_train, is_pos_test = (W_train > 0).astype(int), (W_test > 0).astype(int)
    logreg_W0 = LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, class_weight="balanced"
    ).fit(X_train_np, is_pos_train)
    auc_w0_lr = roc_auc_score(is_pos_test, logreg_W0.predict_proba(X_test_np)[:, 1])

    linreg_W = LinearRegression().fit(X_train_np, W_train)
    W_hat_lr = linreg_W.predict(X_test_np)
    rmse_W_lr, r2_W_lr = np.sqrt(np.mean((W_test - W_hat_lr) ** 2)), r2_score(W_test, W_hat_lr)
    rho_W_lr, _ = spearmanr(W_test, W_hat_lr)

    S_pos_train, S_pos_test = (S_train == 1).astype(int), (S_test == 1).astype(int)
    S_neg_train, S_neg_test = (S_train == -1).astype(int), (S_test == -1).astype(int)

    try:
        auc_Spos_lr = roc_auc_score(
            S_pos_test,
            LogisticRegression(class_weight="balanced").fit(X_train_np, S_pos_train).predict_proba(X_test_np)[:, 1]
        )
    except:
        auc_Spos_lr = np.nan

    try:
        auc_Sneg_lr = roc_auc_score(
            S_neg_test,
            LogisticRegression(class_weight="balanced").fit(X_train_np, S_neg_train).predict_proba(X_test_np)[:, 1]
        )
    except:
        auc_Sneg_lr = np.nan

    # Shuffled baselines (same logic)
    rng = np.random.default_rng(config["seed"] + split_idx)
    X_train_shuff = X_train_np[rng.permutation(X_train_np.shape[0])]
    X_test_shuff = X_test_np[rng.permutation(X_test_np.shape[0])]

    logreg_Y_shuff = LogisticRegression(class_weight="balanced").fit(X_train_shuff, Y_train)
    auc_y_lr_shuff = roc_auc_score(Y_test, logreg_Y_shuff.predict_proba(X_test_shuff)[:, 1])

    logreg_W0_shuff = LogisticRegression(class_weight="balanced").fit(X_train_shuff, is_pos_train)
    auc_w0_lr_shuff = roc_auc_score(is_pos_test, logreg_W0_shuff.predict_proba(X_test_shuff)[:, 1])

    linreg_W_shuff = LinearRegression().fit(X_train_shuff, W_train)
    W_hat_lr_shuff = linreg_W_shuff.predict(X_test_shuff)
    rmse_W_lr_shuff, r2_W_lr_shuff = np.sqrt(np.mean((W_test - W_hat_lr_shuff) ** 2)), r2_score(W_test, W_hat_lr_shuff)
    rho_W_lr_shuff, _ = spearmanr(W_test, W_hat_lr_shuff)

    metrics = {
        "latent_dim": latent_dim,
        "split_idx": split_idx,

        # Baseline NN
        "auc_y_nn": auc_y_nn,
        "auc_w0_nn": auc_w0_nn,
        "rmse_W_nn": rmse_W_nn,
        "r2_W_nn": r2_W_nn,
        "rho_W_nn": rho_W_nn,
        "auc_spos_nn": auc_Spos_nn,
        "auc_sneg_nn": auc_Sneg_nn,

        # VIB Strict single
        "auc_y_vib": auc_y_vib,
        "auc_w0_vib": auc_w0_vib,
        "rmse_W_vib": rmse_W_vib,
        "r2_W_vib": r2_W_vib,
        "rho_W_vib": rho_W_vib,
        "auc_spos_vib": auc_Spos_vib,
        "auc_sneg_vib": auc_Sneg_vib,
        "train_loss_eval_vib": train_loss_eval_vib,
        "test_loss_eval_vib": test_loss_eval_vib,
        "gen_gap_vib": gen_gap_vib,
        "KL_total_vib": KL_total_vib,
        "n_train": n_train,

        # Linear/Logistic Baselines
        "auc_y_lr": auc_y_lr,
        "auc_w0_lr": auc_w0_lr,
        "rmse_W_lr": rmse_W_lr,
        "r2_W_lr": r2_W_lr,
        "rho_W_lr": rho_W_lr,
        "auc_spos_lr": auc_Spos_lr,
        "auc_sneg_lr": auc_Sneg_lr,

        # Shuffled
        "auc_y_lr_shuff": auc_y_lr_shuff,
        "auc_w0_lr_shuff": auc_w0_lr_shuff,
        "rmse_W_lr_shuff": rmse_W_lr_shuff,
        "r2_W_lr_shuff": r2_W_lr_shuff,
        "rho_W_lr_shuff": rho_W_lr_shuff,
    }

    return metrics, info_rows


# ------------------------------------------------------------------
# Main: loop over latent_dim=1..10 and over splits
# ------------------------------------------------------------------
if __name__ == "__main__":
    all_metrics = []
    all_info_rows = []

    latent_dims_to_run = list(range(1, 11))  # 1..10

    for d in latent_dims_to_run:
        print(f"\n\n================== RUNNING latent_dim = {d} ==================\n")
        for k in range(config["n_splits"]):
            m, info_rows = run_one_split(k, latent_dim=d)
            all_metrics.append(m)
            all_info_rows.extend(info_rows)

    # Save metrics over all dims/splits
    df_all = pd.DataFrame(all_metrics)
    metrics_all_path = os.path.join(OUTPUT_DIR, "metrics_all_latdims.csv")
    df_all.to_csv(metrics_all_path, index=False)
    print(f"\nSaved: {metrics_all_path}")

    # Save info-theory traces
    df_info = pd.DataFrame(all_info_rows)
    info_path = os.path.join(OUTPUT_DIR, "info_theory_trace_latdims.csv")
    df_info.to_csv(info_path, index=False)
    print(f"Saved: {info_path}")

    # Summary table: mean ± std per latent_dim
    print("\n================ Summary by latent_dim ================")
    summary_rows = []
    metric_cols = [c for c in df_all.columns if c not in ("split_idx", "latent_dim")]

    for d in latent_dims_to_run:
        sub = df_all[df_all["latent_dim"] == d]
        row = {"latent_dim": d}
        for col in metric_cols:
            vals = sub[col].astype(float).to_numpy()
            row[col + "_mean"] = np.nanmean(vals)
            row[col + "_std"] = np.nanstd(vals)
        summary_rows.append(row)

        # print a compact summary for AUC_y_vib and R2_w_vib for quick sanity
        print(
            f"latent_dim={d}: "
            f"auc_y_vib={row['auc_y_vib_mean']:.3f}±{row['auc_y_vib_std']:.3f} | "
            f"r2_W_vib={row['r2_W_vib_mean']:.3f}±{row['r2_W_vib_std']:.3f} | "
            f"auc_y_nn={row['auc_y_nn_mean']:.3f}±{row['auc_y_nn_std']:.3f}"
        )

    df_summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUTPUT_DIR, "metrics_summary_latdims.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"\nSaved: {summary_path}")