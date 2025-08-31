#!/usr/bin/env python3
"""
prepare_dataset.py — CSV-only, schema-compatible (columns/categoricals/numericals), capped rows

Supports:
  --dataset {home_credit, paysim, generic}
  --path     Path to raw CSV (Home Credit: application_train.csv; PaySim: PS_*.csv; Generic: CSV/Parquet)
  --max_rows Max rows to read from the primary file (default: 50_000)
Outputs (CSV only):
  <out_dir>/train.csv
  <out_dir>/valid.csv
  <out_dir>/schema.json
  <out_dir>/eda_report.md

Notes:
- Home Credit: keeps both DEFAULT and TARGET (= DEFAULT). Adds simple per-SK_ID_CURR counts from other files if available.
- PaySim: keeps DEFAULT (= isFraud) and isFraud, drops high-cardinality IDs, adds HOUR from step.
- Generic: keeps target, infers schema.
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
    repl = {t: np.nan for t in MISSING_TOKENS}
    for c in out.columns:
        out[c] = out[c].replace(repl)
        if out[c].dtype == object:
            out[c] = out[c].astype(str).str.strip().replace(repl)
    return out

def _drop_bad_columns(df: pd.DataFrame, target: str, max_missing=0.95, max_single_value=0.99) -> pd.DataFrame:
    drop_cols = []
    n = len(df)
    for c in df.columns:
        if c == target:  # never drop target
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

def _infer_types(df: pd.DataFrame, target: Optional[str]) -> Tuple[List[str], List[str]]:
    cats, nums = [], []
    for c in df.columns:
        if target is not None and c == target:
            continue
        if np.issubdtype(pd.Series(df[c]).dtype, np.number):
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
        uniq = df[target].dropna().unique()
        if set(pd.Series(uniq).dropna().unique()).issubset({0,1}):
            pos = int((df[target]==1).sum()); neg = int((df[target]==0).sum())
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
# Home Credit extras (optional file joins)
# -----------------------------
def _count_by_id(csv_path: Path, id_col: str, id_filter: pd.Series) -> Optional[pd.DataFrame]:
    """Return dataframe with counts per id_col for ids in id_filter; None if file missing."""
    if not csv_path.exists():
        return None
    # Read only the id column to keep memory low
    try:
        s = pd.read_csv(csv_path, usecols=[id_col], low_memory=True)
    except Exception:
        return None
    s = s[id_col].dropna()
    vc = s.value_counts()
    vc = vc[vc.index.isin(set(id_filter.values))]
    out = vc.rename("COUNT").reset_index().rename(columns={"index": id_col})
    return out

def load_home_credit(app_train_path: str, max_rows: int) -> Tuple[pd.DataFrame, str, List[str], List[str], List[str]]:
    app_train_path = Path(app_train_path)
    root = app_train_path.parent

    # Read first N rows of application_train.csv
    base_cols = [
        "SK_ID_CURR", "TARGET",
        "CODE_GENDER","FLAG_OWN_CAR","FLAG_OWN_REALTY","CNT_CHILDREN",
        "AMT_INCOME_TOTAL","AMT_CREDIT","AMT_ANNUITY",
        "NAME_INCOME_TYPE","NAME_EDUCATION_TYPE","NAME_FAMILY_STATUS","NAME_HOUSING_TYPE",
        "DAYS_BIRTH","DAYS_EMPLOYED","OWN_CAR_AGE","OCCUPATION_TYPE",
        "REGION_RATING_CLIENT","REGION_RATING_CLIENT_W_CITY",
        "EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3"
    ]
    usecols = [c for c in base_cols if c != "OWN_CAR_AGE"]  # OWN_CAR_AGE may be absent in some rows; add later if present
    df = pd.read_csv(app_train_path, nrows=max_rows, low_memory=True, usecols=lambda c: True)

    # Ensure we only keep requested columns (if present)
    keep = [c for c in base_cols if c in df.columns]
    df = df[keep].copy()

    # Rename TARGET -> DEFAULT and duplicate TARGET
    if "TARGET" not in df.columns:
        raise ValueError("Home Credit file missing 'TARGET'")
    df = df.rename(columns={"TARGET": "DEFAULT"})
    df["TARGET"] = df["DEFAULT"]

    # Optional: merge simple counts from auxiliary files using SK_ID_CURR
    if "SK_ID_CURR" in df.columns:
        ids = df["SK_ID_CURR"]

        extra_feats = []
        # bureau.csv
        b = _count_by_id(root / "bureau.csv", "SK_ID_CURR", ids)
        if b is not None:
            b = b.rename(columns={"COUNT": "BUREAU_COUNT"})
            df = df.merge(b, on="SK_ID_CURR", how="left")
            extra_feats.append("BUREAU_COUNT")
        # previous_application.csv
        p = _count_by_id(root / "previous_application.csv", "SK_ID_CURR", ids)
        if p is not None:
            p = p.rename(columns={"COUNT": "PREVAPP_COUNT"})
            df = df.merge(p, on="SK_ID_CURR", how="left")
            extra_feats.append("PREVAPP_COUNT")
        # POS_CASH_balance.csv
        pos = _count_by_id(root / "POS_CASH_balance.csv", "SK_ID_CURR", ids)
        if pos is not None:
            pos = pos.rename(columns={"COUNT": "POS_COUNT"})
            df = df.merge(pos, on="SK_ID_CURR", how="left")
            extra_feats.append("POS_COUNT")
        # installments_payments.csv
        inst = _count_by_id(root / "installments_payments.csv", "SK_ID_CURR", ids)
        if inst is not None:
            inst = inst.rename(columns={"COUNT": "INST_COUNT"})
            df = df.merge(inst, on="SK_ID_CURR", how="left")
            extra_feats.append("INST_COUNT")
        # credit_card_balance.csv
        cc = _count_by_id(root / "credit_card_balance.csv", "SK_ID_CURR", ids)
        if cc is not None:
            cc = cc.rename(columns={"COUNT": "CC_COUNT"})
            df = df.merge(cc, on="SK_ID_CURR", how="left")
            extra_feats.append("CC_COUNT")
    else:
        extra_feats = []

    # Drop SK_ID_CURR from modeling set
    if "SK_ID_CURR" in df.columns:
        df = df.drop(columns=["SK_ID_CURR"])

    # Schema lists based on your provided schema + added extras
    # Categorical list from your schema
    categoricals = [
        "CODE_GENDER","FLAG_OWN_CAR","FLAG_OWN_REALTY",
        "NAME_INCOME_TYPE","NAME_EDUCATION_TYPE","NAME_FAMILY_STATUS",
        "NAME_HOUSING_TYPE","OCCUPATION_TYPE"
    ]
    # Numerical list from your schema + aggregates we created (if any)
    numericals = [
        "TARGET","CNT_CHILDREN","AMT_INCOME_TOTAL","AMT_CREDIT","AMT_ANNUITY",
        "DAYS_BIRTH","DAYS_EMPLOYED","OWN_CAR_AGE","REGION_RATING_CLIENT",
        "REGION_RATING_CLIENT_W_CITY","EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3"
    ] + extra_feats
    # Columns order (match your example: DEFAULT, TARGET, then categoricals, then numericals excluding TARGET to avoid duplicate)
    columns = ["DEFAULT","TARGET"] + categoricals + [c for c in numericals if c != "TARGET"]

    # Keep only columns that exist
    columns = [c for c in columns if c in df.columns]
    categoricals = [c for c in categoricals if c in df.columns]
    numericals = [c for c in numericals if c in df.columns]

    # Reorder df to columns order
    df = df.reindex(columns=columns)

    return df, "DEFAULT", columns, categoricals, numericals

def load_paysim(csv_path: str, max_rows: int) -> Tuple[pd.DataFrame, str, List[str], List[str], List[str]]:
    df = pd.read_csv(csv_path, nrows=max_rows, low_memory=True)
    if "isFraud" not in df.columns:
        raise ValueError("PaySim file missing 'isFraud'")
    df["DEFAULT"] = df["isFraud"]

    # Drop high-cardinality IDs
    for c in ["nameOrig","nameDest"]:
        if c in df.columns:
            df = df.drop(columns=c)

    # HOUR from step
    if "step" in df.columns and "HOUR" not in df.columns:
        df["HOUR"] = (df["step"] % 24).astype(int)

    # Arrange columns to match your schema
    columns = [
        "DEFAULT","type","amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest","HOUR"
    ]
    columns = [c for c in columns if c in df.columns]
    df = df[columns].copy()

    categoricals = [c for c in ["type"] if c in df.columns]
    numericals   = [c for c in ["amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest","HOUR"] if c in df.columns]

    return df, "DEFAULT", columns, categoricals, numericals

def load_generic(path: str, target: Optional[str], max_rows: int) -> Tuple[pd.DataFrame, str, List[str], List[str], List[str]]:
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
        if max_rows and len(df) > max_rows:
            df = df.head(max_rows)
    else:
        df = pd.read_csv(p, nrows=max_rows, low_memory=True)
    if not target:
        raise ValueError("--target must be provided for generic dataset.")
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in file.")

    # Infer basic schema
    cats, nums = _infer_types(df, target)
    # Column order: target first, then cats, then nums
    columns = [target] + cats + nums
    df = df.reindex(columns=columns)

    return df, target, columns, cats, nums

# -----------------------------
# Clean, split, write
# -----------------------------
def clean_and_split(df: pd.DataFrame, target: str, test_size=0.2, seed=42):
    df = _normalize_missing(df)
    df = _drop_bad_columns(df, target)
    cats, nums = _infer_types(df, target)
    df = _winsorize(df, nums)

    # Stratified split if binary
    strat = df[target] if set(pd.Series(df[target]).dropna().unique()).issubset({0,1}) else None
    train_df, valid_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=strat)

    # Re-infer on train for schema (robustness after drops)
    cats, nums = _infer_types(train_df, target)

    return train_df, valid_df, cats, nums

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["home_credit","paysim","generic"], required=True)
    ap.add_argument("--path", required=True, help="Home Credit: application_train.csv; PaySim: main CSV; Generic: CSV/Parquet.")
    ap.add_argument("--target", default=None, help="Target column for generic dataset.")
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_rows", type=int, default=50_000)
    args = ap.parse_args()

    if args.dataset == "home_credit":
        df, target, columns, cats, nums = load_home_credit(args.path, args.max_rows)
    elif args.dataset == "paysim":
        df, target, columns, cats, nums = load_paysim(args.path, args.max_rows)
    else:
        df, target, columns, cats, nums = load_generic(args.path, args.target, args.max_rows)

    # Clean & split (preserves target)
    train_df, valid_df, cats2, nums2 = clean_and_split(df, target, args.test_size, args.seed)

    # Schema: keep your desired layout ("columns") and also include "target" for downstream code
    # Reconcile with post-cleaning availability
    available_cols = list(train_df.columns)
    columns_final = [c for c in columns if c in available_cols]
    cats_final = [c for c in cats if c in available_cols]
    nums_final = [c for c in nums if c in available_cols]

    schema = {
        "target": target,                 # keep for Transformer compatibility
        "columns": columns_final,         # explicit column order for generation
        "categoricals": cats_final,
        "numericals": nums_final,
        "notes": {
            "winsorized_quantiles": [0.005, 0.995],
            "format": "csv"
        }
    }

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    train_path = out / "train.csv"
    valid_path = out / "valid.csv"
    schema_path = out / "schema.json"
    eda_path = out / "eda_report.md"

    train_df.to_csv(train_path, index=False)
    valid_df.to_csv(valid_path, index=False)
    schema_path.write_text(json.dumps(schema, indent=2))

    eda_md = _profile(train_df, target, cats_final, nums_final)
    eda_path.write_text(eda_md)

    print("Saved:")
    print(f"- {train_path}")
    print(f"- {valid_path}")
    print(f"- {schema_path}")
    print(f"- {eda_path}")

if __name__ == "__main__":
    main()
