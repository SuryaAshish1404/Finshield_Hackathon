#!/usr/bin/env python3
"""
credit_model.py  — NN classifier with leakage-safe feature selection + probability calibration

Trains a simple feed-forward NN to predict DEFAULT (0/1) using schema.json for feature sets.
Removes label-like columns from features (DEFAULT, TARGET, isFraud) to prevent leakage.
Applies isotonic calibration on validation predictions and uses it at scoring time.

Artifacts (under models/<dataset_name>/):
  - credit_nn.pt                    (PyTorch weights)
  - credit_nn_preproc.joblib        (sklearn preprocessors + feature order)
  - prob_calibrator.joblib          (isotonic calibration; optional)
  - score_meta.json                 (credit score mapping doc)

Usage:
  # Train
  python credit_model.py train \
    --dataset_name home_credit \
    --data_dir data/processed/home_credit \
    --models_dir models \
    --epochs 20 --batch_size 512 --patience 5

  # Score
  python credit_model.py score \
    --dataset_name home_credit \
    --models_dir models \
    --input_csv data/processed/home_credit/imputed_rows.csv \
    --out_csv data/processed/home_credit/scored_nn.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from sklearn.isotonic import IsotonicRegression


# --------------------------
# Helpers
# --------------------------
def load_schema(data_dir: Path):
    schema = json.loads((data_dir / "schema.json").read_text(encoding="utf-8"))
    target = schema["target"]
    cols = schema.get("columns", schema["categoricals"] + schema["numericals"])
    cats = schema["categoricals"]
    nums = schema["numericals"]
    return target, cols, cats, nums


def load_train_valid(data_dir: Path):
    train = pd.read_csv(data_dir / "train.csv")
    valid_path = data_dir / "valid.csv"
    valid = pd.read_csv(valid_path) if valid_path.exists() else None
    return train, valid


def probability_to_score(prob_no_default: float, lo=300, hi=850) -> int:
    score = lo + int(np.clip(prob_no_default, 0.0, 1.0) * (hi - lo))
    return int(np.clip(score, lo, hi))


def build_preprocessor(cat_cols, num_cols):
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True))
    ])
    pre = ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, cat_cols),
            ("num", num_pipe, num_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre


class TensorDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = None if y is None else torch.tensor(y.astype(np.float32))
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i):
        if self.y is None:
            return self.X[i]
        return self.X[i], self.y[i]


class MLP(nn.Module):
    def __init__(self, in_dim, hidden=(256,128), p=0.2):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(p)]
            last = h
        layers += [nn.Linear(last, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):  # returns logits
        return self.net(x).squeeze(1)


def train_nn(X_train, y_train, X_valid, y_valid, epochs=20, batch_size=512, patience=5, weight_decay=1e-4, device="cpu"):
    train_ds = TensorDataset(X_train, y_train)
    valid_ds = TensorDataset(X_valid, y_valid) if X_valid is not None else None

    in_dim = X_train.shape[1]
    model = MLP(in_dim, hidden=(256,128), p=0.3).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    crit = nn.BCEWithLogitsLoss()
    best_auc, best_state, wait = -1.0, None, 0

    for epoch in range(1, epochs+1):
        # ---- train
        model.train()
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = crit(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()

        # ---- evaluate
        def _eval(X, y):
            dl = DataLoader(TensorDataset(X, y), batch_size=4096, shuffle=False)
            model.eval()
            preds, ys = [], []
            with torch.no_grad():
                for xb, yb in dl:
                    logits = model(xb.to(device))
                    preds.append(torch.sigmoid(logits).cpu().numpy())
                    ys.append(yb.numpy())
            p = np.concatenate(preds); yy = np.concatenate(ys).astype(int)
            auc = roc_auc_score(yy, p) if len(np.unique(yy)) > 1 else np.nan
            acc = accuracy_score(yy, (p >= 0.5).astype(int))
            ll = log_loss(yy, p, labels=[0,1])
            return p, auc, acc, ll

        p_tr, auc_tr, acc_tr, ll_tr = _eval(X_train, y_train)
        if X_valid is not None:
            p_va, auc_va, acc_va, ll_va = _eval(X_valid, y_valid)
            print(f"Epoch {epoch:02d}  train_loss={total_loss/len(loader):.4f} | "
                  f"train AUC={auc_tr:.4f} Acc={acc_tr:.4f} | valid AUC={auc_va:.4f} Acc={acc_va:.4f}")
            # early stopping on valid AUC
            metric = 0.0 if np.isnan(auc_va) else auc_va
            if metric > best_auc:
                best_auc, best_state, wait = metric, {k:v.cpu() for k,v in model.state_dict().items()}, 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping (patience={patience}). Best AUC={best_auc:.4f}")
                    break
        else:
            print(f"Epoch {epoch:02d}  train_loss={total_loss/len(loader):.4f} | train AUC={auc_tr:.4f} Acc={acc_tr:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def leakage_safe_columns(ordered_cols, df, target):
    label_like = {target, "DEFAULT", "TARGET", "isFraud"}
    return [c for c in ordered_cols if (c in df.columns and c not in label_like)]


# --------------------------
# Commands
# --------------------------
def cmd_train(args):
    data_dir = Path(args.data_dir)
    mdl_dir = Path(args.models_dir) / args.dataset_name
    mdl_dir.mkdir(parents=True, exist_ok=True)

    target, ordered_cols, cats, nums = load_schema(data_dir)
    train_df, valid_df = load_train_valid(data_dir)

    if target not in train_df.columns:
        raise ValueError(f"Target '{target}' not in train.csv")

    # Leakage-safe feature selection (drops DEFAULT/TARGET/isFraud if present)
    used_cols = leakage_safe_columns(ordered_cols, train_df, target)
    cats_used = [c for c in cats if c in used_cols]
    nums_used = [n for n in nums if n in used_cols]

    pre = build_preprocessor(cats_used, nums_used)

    X_train = train_df[used_cols].copy()
    y_train = train_df[target].astype(int).values

    if valid_df is not None and target in valid_df.columns:
        X_valid = valid_df[used_cols].copy()
        y_valid = valid_df[target].astype(int).values
    else:
        X_valid = None
        y_valid = None

    # Fit preprocessor on train only
    X_train_enc = pre.fit_transform(X_train)
    X_valid_enc = pre.transform(X_valid) if X_valid is not None else None

    # Persist preprocessor + feature lists
    joblib.dump({
        "pre": pre,
        "feature_order": used_cols,
        "categoricals": cats_used,
        "numericals": nums_used,
        "target": target
    }, mdl_dir / "credit_nn_preproc.joblib")

    # Train NN
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = train_nn(
        X_train_enc, y_train,
        X_valid_enc, y_valid,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        weight_decay=1e-4,
        device=device
    )

    # Save model weights
    torch.save(model.state_dict(), mdl_dir / "credit_nn.pt")
    print(f"✔ Saved model → {mdl_dir/'credit_nn.pt'}")

    # ---- Validation metrics + calibration
    if X_valid_enc is not None:
        model.eval()
        with torch.no_grad():
            p_val = torch.sigmoid(model(torch.tensor(X_valid_enc, dtype=torch.float32).to(device))).cpu().numpy()
        auc = roc_auc_score(y_valid, p_val) if len(np.unique(y_valid)) > 1 else np.nan
        ll = log_loss(y_valid, p_val, labels=[0,1])
        acc = accuracy_score(y_valid, (p_val >= 0.5).astype(int))
        print(f"[valid] AUC={auc:.4f}  LogLoss={ll:.4f}  Acc={acc:.4f}")

        # Isotonic calibration on validation
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(p_val, y_valid)
        joblib.dump(calibrator, mdl_dir / "prob_calibrator.joblib")
        print(f"✔ Saved calibrator → {mdl_dir/'prob_calibrator.joblib'}")
    else:
        print("No valid.csv → skipping calibration.")
    
    # Save score mapping info
    score_meta = {
        "score_scale": {"min": 300, "max": 850},
        "mapping": "score = 300 + round(P(no default) * 550)",
        "target": target,
        "dataset_name": args.dataset_name,
    }
    (mdl_dir / "score_meta.json").write_text(json.dumps(score_meta, indent=2))
    print(f"✔ Saved score meta → {mdl_dir/'score_meta.json'}")


def cmd_score(args):
    mdl_dir = Path(args.models_dir) / args.dataset_name
    bundle = joblib.load(mdl_dir / "credit_nn_preproc.joblib")
    pre = bundle["pre"]
    used_cols = bundle["feature_order"]
    target = bundle["target"]

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Infer input size from pre.transform one dummy row if needed
    dummy = pre.transform(pd.DataFrame([ {c: np.nan for c in used_cols} ]))
    in_dim = dummy.shape[1]
    model = MLP(in_dim, hidden=(256,128), p=0.3).to(device)
    model.load_state_dict(torch.load(mdl_dir / "credit_nn.pt", map_location="cpu"))
    model.eval()

    # Optional calibrator
    cal_path = mdl_dir / "prob_calibrator.joblib"
    calibrator = joblib.load(cal_path) if cal_path.exists() else None

    # Read input and align columns
    df = pd.read_csv(args.input_csv)
    X = df.copy()
    for c in used_cols:
        if c not in X.columns:
            X[c] = np.nan
    X = X[used_cols]

    X_enc = pre.transform(X)
    with torch.no_grad():
        p_def = torch.sigmoid(
            model(torch.tensor(X_enc, dtype=torch.float32).to(device))
        ).cpu().numpy()

    if calibrator is not None:
        p_def = calibrator.predict(p_def)

    p_no_def = 1.0 - p_def
    scores = [probability_to_score(p) for p in p_no_def]

    out = df.copy()
    out["prob_default"] = p_def
    out["prob_no_default"] = p_no_def
    out["credit_score"] = scores

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"✔ Scored {len(out)} rows → {args.out_csv}")


# --------------------------
# Main
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train", help="Train NN credit model")
    tr.add_argument("--dataset_name", required=True, choices=["home_credit","paysim","generic"])
    tr.add_argument("--data_dir", required=True)
    tr.add_argument("--models_dir", default="models")
    tr.add_argument("--epochs", type=int, default=20)
    tr.add_argument("--batch_size", type=int, default=512)
    tr.add_argument("--patience", type=int, default=5)

    sc = sub.add_parser("score", help="Score rows with trained NN model")
    sc.add_argument("--dataset_name", required=True, choices=["home_credit","paysim","generic"])
    sc.add_argument("--models_dir", default="models")
    sc.add_argument("--input_csv", required=True)
    sc.add_argument("--out_csv", required=True)

    args = ap.parse_args()
    if args.cmd == "train":
        cmd_train(args)
    else:
        cmd_score(args)


if __name__ == "__main__":
    main()
