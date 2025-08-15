#!/usr/bin/env python3
"""
prepare_dataset.py  — CSV-only pipeline

- Supports: --dataset {home_credit, paysim, generic}
- Input: a raw CSV (or Parquet for generic if you really want; we still prefer CSV)
- Output (CSV-only):
    data/processed/train.csv
    data/processed/valid.csv
    data/processed/schema.json
    data/processed/eda_report.md

Notes:
- We KEEP label columns in the outputs. For PaySim we keep BOTH `isFraud` and `DEFAULT` (DEFAULT=isFraud).
- We infer simple schema (categoricals vs numericals) and DO NOT drop the target.
- Light cleaning: normalize common missing tokens, drop awful columns, winsorize numerics, optional HOUR feature for PaySim.
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# -----------------------------
# Helpers
# -----------------------------
MISSING_TOKENS = {"", "na", "n/a", "none", "null", "unknown", "xna", "nan", "NaN", "None"}

def _normalize_missing(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        # treat stringy missings as NaN
        out[c] = out[c].replace({t: np.nan for t in MISSING_TOKENS})
        if out[c].dtype == object:
            out[c] = (
                out[c]
                .astype(str)
                .str.strip()
                .replace({t: np.nan for t in MISSING_TOKENS})
            )
    return out

def _drop_bad_columns(df: pd.DataFrame, target: str, max_missing=0.95, max_single_value=0.99) -> pd.DataFrame:
    drop_cols = []
    n = len(df)
    for c in df.columns:
        if c == target:
            continue
        miss = df[c].isna().mean()
        if miss > max_missing:
            drop_cols.append(c); continue
        vc = df[c].value_counts(dropna=True)
        if len(vc) > 0:
            top_ratio = vc.iloc[0] / max(1, n - df[c].isna().sum())
            if top_ratio > max_single_value:
                drop_cols.append(c)
    return df.drop(columns=drop_cols) if drop_cols else df

def _infer_types(df: pd.DataFrame, target: str) -> Tuple[List[str], List[str]]:
    cats, nums = [], []
    for c in df.columns:
        if c == target:
            continue
        if np.issubdtype(df[c].dtype, np.number):
            nums.append(c)
        else:
            cats.append(c)
    return cats, nums

def _winsorize(df: pd.DataFrame, nums: List[str], lo=0.005, hi=0.995) -> pd.DataFrame:
    out = df.copy()
    for c in nums:
        x = pd.to_numeric(out[c], errors="coerce")
        ql, qh = x.quantile(lo), x.quantile(hi)
        out[c] = x.clip(ql, qh)
    return out

def _profile(df: pd.DataFrame, target: str, cats: List[str], nums: List[str]) -> str:
    lines = []
    lines.append("# EDA Summary")
    lines.append(f"- Rows (train): {len(df)} | Cols: {df.shape[1]}")
    if target in df.columns:
        if set(df[target].dropna().unique()).issubset({0,1}):
            pos = int((df[target]==1).sum())
            neg = int((df[target]==0).sum())
            lines.append(f"- Target '{target}' (train): 1s={pos} ({pos/len(df):.2%}), 0s={neg} ({neg/len(df):.2%})")
        else:
            lines.append(f"- Target '{target}' is non-binary (unique={df[target].nunique()})")
    miss = df.isna().mean().sort_values(ascending=False).head(15)
    if len(miss):
        lines.append("\n## Top-15 Missingness (train)")
        lines.append(miss.to_string())
    lines.append("\n## Column Types")
    lines.append(f"- Categorical ({len(cats)}): {cats[:30]}{' ...' if len(cats)>30 else ''}")
    lines.append(f"- Numerical ({len(nums)}): {nums[:30]}{' ...' if len(nums)>30 else ''}")
    return "\n".join(lines)


# -----------------------------
# Dataset adapters
# -----------------------------
def load_home_credit(path: str) -> Tuple[pd.DataFrame, str]:
    # Expect application_train.csv (huge CSV)
    df = pd.read_csv(path)
    if "TARGET" not in df.columns:
        raise ValueError("Home Credit file missing 'TARGET'")
    # Keep label as DEFAULT (we also keep original TARGET for reference)
    df = df.rename(columns={"TARGET": "DEFAULT"})
    df["TARGET"] = df["DEFAULT"]
    keep = [
        "DEFAULT", "TARGET",
        "CODE_GENDER","FLAG_OWN_CAR","FLAG_OWN_REALTY","CNT_CHILDREN",
        "AMT_INCOME_TOTAL","AMT_CREDIT","AMT_ANNUITY",
        "NAME_INCOME_TYPE","NAME_EDUCATION_TYPE","NAME_FAMILY_STATUS","NAME_HOUSING_TYPE",
        "DAYS_BIRTH","DAYS_EMPLOYED","OWN_CAR_AGE","OCCUPATION_TYPE",
        "REGION_RATING_CLIENT","REGION_RATING_CLIENT_W_CITY",
        "EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3"
    ]
    df = df[[c for c in keep if c in df.columns]].copy()
    return df, "DEFAULT"

def load_paysim(path: str) -> Tuple[pd.DataFrame, str]:
    # PaySim: we KEEP isFraud and add DEFAULT = isFraud
    df = pd.read_csv(path)
    if "isFraud" not in df.columns:
        raise ValueError("PaySim file missing 'isFraud'")
    df["DEFAULT"] = df["isFraud"]
    # Drop high-cardinality IDs
    for c in ["nameOrig","nameDest"]:
        if c in df.columns:
            df = df.drop(columns=c)
    # Derive hour-of-day from step (each step ~1 hour)
    if "step" in df.columns:
        df["HOUR"] = (df["step"] % 24).astype(int)
    keep = [
        "DEFAULT","isFraud","type","amount",
        "oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest",
        "HOUR"
    ]
    df = df[[c for c in keep if c in df.columns]].copy()
    return df, "DEFAULT"

def load_generic(path: str, target: Optional[str]) -> Tuple[pd.DataFrame, str]:
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)  # we still allow reading Parquet if you already have one
    else:
        df = pd.read_csv(path)
    if not target:
        raise ValueError("--target must be provided for generic dataset.")
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in file.")
    return df, target


# -----------------------------
# Main cleaning & split
# -----------------------------
def clean_and_split(df: pd.DataFrame, target: str, test_size=0.2, seed=42):
    df = _normalize_missing(df)
    df = _drop_bad_columns(df, target)
    cats, nums = _infer_types(df, target)
    df = _winsorize(df, nums)

    # Stratified split if target looks binary
    strat = df[target] if set(df[target].dropna().unique()).issubset({0,1}) else None
    train_df, valid_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=strat)

    # Re-infer on train for schema
    cats, nums = _infer_types(train_df, target)
    schema = {
        "target": target,
        "categoricals": cats,
        "numericals": nums,
        "notes": {
            "winsorized_quantiles": [0.005, 0.995],
            "format": "csv"
        }
    }
    return train_df, valid_df, schema


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["home_credit","paysim","generic"], required=True)
    ap.add_argument("--path", required=True, help="Path to raw CSV (or Parquet for generic).")
    ap.add_argument("--target", default=None, help="Generic dataset target column.")
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.dataset == "home_credit":
        df, target = load_home_credit(args.path)
    elif args.dataset == "paysim":
        df, target = load_paysim(args.path)
    else:
        df, target = load_generic(args.path, args.target)

    train_df, valid_df, schema = clean_and_split(df, target, args.test_size, args.seed)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # ---- CSV ONLY ----
    train_path = out / "train.csv"
    valid_path = out / "valid.csv"
    schema_path = out / "schema.json"
    eda_path = out / "eda_report.md"

    train_df.to_csv(train_path, index=False)
    valid_df.to_csv(valid_path, index=False)
    schema_path.write_text(json.dumps(schema, indent=2))

    eda_md = _profile(train_df, target, schema["categoricals"], schema["numericals"])
    eda_path.write_text(eda_md)

    print("Saved:")
    print(f"- {train_path}")
    print(f"- {valid_path}")
    print(f"- {schema_path}")
    print(f"- {eda_path}")

if __name__ == "__main__":
    main()
