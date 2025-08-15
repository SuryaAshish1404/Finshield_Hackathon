# Evaluation.py — schema-aware fidelity checks & simple detectability test
# Reads:  real=data/processed/train.parquet (or valid.parquet), synthetic_output.csv
# Writes: reports/plots/*.png and reports/synth_eval.json

import argparse, json, os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp

def numeric_ks(real, synth, cols):
    ks = {}
    for c in cols:
        r = pd.to_numeric(real[c], errors="coerce").dropna()
        s = pd.to_numeric(synth[c], errors="coerce").dropna()
        if len(r) > 10 and len(s) > 10:
            ks[c] = float(ks_2samp(r, s).statistic)
    return ks

def plot_distributions(real, synth, cats, nums, outdir):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    for c in nums:
        plt.figure(figsize=(5,3.2))
        sns.kdeplot(pd.to_numeric(real[c], errors="coerce").dropna(), label="Real", fill=True)
        sns.kdeplot(pd.to_numeric(synth[c], errors="coerce").dropna(), label="Synthetic", fill=True, linestyle="--")
        plt.title(f"{c} (numeric)"); plt.tight_layout()
        plt.legend(); plt.savefig(f"{outdir}/{c}_num.png"); plt.close()
    for c in cats:
        rc = real[c].astype(str).value_counts(normalize=True)
        sc = synth[c].astype(str).value_counts(normalize=True)
        df = pd.DataFrame({"Real": rc, "Synthetic": sc}).fillna(0)
        ax = df.plot(kind="bar", figsize=(6,3.2))
        ax.set_title(f"{c} (categorical)"); plt.tight_layout()
        plt.savefig(f"{outdir}/{c}_cat.png"); plt.close()

def detectability_auc(real, synth, cats, nums):
    realX = real[cats+nums].copy(); synthX = synth[cats+nums].copy()
    realX["__label__"] = 1; synthX["__label__"] = 0
    df = pd.concat([realX, synthX], axis=0, ignore_index=True).dropna()
    y = df["__label__"].values; X = df.drop(columns=["__label__"])
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cats),
                             ("num", StandardScaler(), nums)], remainder="drop")
    clf = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=500))])
    clf.fit(X, y)
    p = clf.predict_proba(X)[:,1]
    return float(roc_auc_score(y, p))

def main(args):
    schema = json.loads(Path(args.schema).read_text())
    cats, nums = schema["categoricals"], schema["numericals"]

    real = pd.read_parquet(args.real)
    synth = pd.read_csv(args.synthetic) if args.synthetic.endswith(".csv") else pd.read_parquet(args.synthetic)

    # Align columns and types
    for c in cats+nums:
        if c not in synth.columns:
            synth[c] = np.nan
    synth = synth[cats+nums]

    ks = numeric_ks(real, synth, nums)
    plot_distributions(real, synth, cats, nums, args.plots_dir)
    auc = detectability_auc(real, synth, cats, nums)

    out = {
        "numeric_KS_by_column": ks,
        "mean_numeric_KS": float(np.mean(list(ks.values()))) if ks else None,
        "detectability_auc": auc  # closer to 0.5 is better (hard to tell real vs synthetic)
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"Wrote: {args.out_json} and plots in {args.plots_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default="data/processed/train.parquet")
    ap.add_argument("--synthetic", default="data/processed/synthetic_output.csv")
    ap.add_argument("--schema", default="data/processed/schema.json")
    ap.add_argument("--plots_dir", default="reports/plots")
    ap.add_argument("--out_json", default="reports/synth_eval.json")
    args = ap.parse_args()
    main(args)
