#!/usr/bin/env python3
"""
Impute.py — fill missing values in rows using your trained tabular generator.

- Loads GPTMini/SimpleTokenizer/generate_synthetic_row from your training notebook (.ipynb)
- Loads artifacts from models/<dataset_name>/:
    * <dataset_name>_tokenizer.pkl
    * <dataset_name>_transformer_model.pt   (or <dataset_name>_transformer.pt)
    * meta.json  (contains schema, token_maps, numeric encoding, column_order)

Usage (PowerShell-friendly):
python Impute.py `
  --dataset_name home_credit `
  --models_dir models `
  --nb_path HOME-CREDITED-Transformer.ipynb `
  --input_csv data/processed/home_credit/train.csv `
  --out data/processed/home_credit/imputed_rows.csv `
  --max_rows 100
"""
import argparse
import json
import pickle
import types
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm


# -----------------------------
# Load a .ipynb as a Python module and seed schema globals
# -----------------------------
def load_notebook_module(nb_path: Path, schema: dict, module_name: str = "finshield_nb") -> types.ModuleType:
    import nbformat
    from IPython.core.interactiveshell import InteractiveShell

    nb = nbformat.read(str(nb_path), as_version=4)
    module = types.ModuleType(module_name)
    module.__dict__["__file__"] = str(nb_path)

    # Seed schema-derived globals so notebook cells that reference them won't crash
    module.__dict__["categorical_columns"] = list(schema.get("categoricals", []))
    module.__dict__["numerical_columns"] = list(schema.get("numericals", []))
    module.__dict__["column_order"] = list(schema.get("columns", module.__dict__["categorical_columns"] + module.__dict__["numerical_columns"]))
    module.__dict__["token_maps"] = schema.get("token_maps", {})  # may be updated by meta.json later

    shell = InteractiveShell.instance()
    exec_env = module.__dict__

    for cell in nb.cells:
        if cell.cell_type == "code":
            code = shell.input_transformer_manager.transform_cell(cell.source)
            exec(code, exec_env)

    # Ensure the required symbols exist
    for name in ("GPTMini", "SimpleTokenizer", "generate_synthetic_row"):
        if not hasattr(module, name):
            raise AttributeError(f"Notebook import failed: missing `{name}` after import.")
    print("✅ Exported: GPTMini, SimpleTokenizer, generate_synthetic_row, schema globals")
    return module


# -----------------------------
# Robust tokenizer unpickling (map to the notebook's SimpleTokenizer if needed)
# -----------------------------
class _RenameUnpickler(pickle.Unpickler):
    def __init__(self, file, simple_tokenizer_cls=None):
        super().__init__(file)
        self._st_cls = simple_tokenizer_cls

    def find_class(self, module, name):
        if name == "SimpleTokenizer" and self._st_cls is not None:
            return self._st_cls
        return super().find_class(module, name)


def _load_tokenizer(path: Path, simple_tokenizer_cls=None):
    """Load tokenizer; if pickled against another module, remap to notebook class."""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except AttributeError:
        with open(path, "rb") as f:
            return _RenameUnpickler(f, simple_tokenizer_cls).load()


# -----------------------------
# Artifacts helpers
# -----------------------------
def resolve_artifact_paths(model_dir: Path, dataset_name: str):
    tok_path = model_dir / f"{dataset_name}_tokenizer.pkl"
    meta_path = model_dir / "meta.json"
    model_candidates = [
        model_dir / f"{dataset_name}_transformer_model.pt",
        model_dir / f"{dataset_name}_transformer.pt",
    ]
    mdl_path = next((p for p in model_candidates if p.exists()), None)
    return tok_path, mdl_path, meta_path


def sync_globals_from_meta(nbmod, meta: dict):
    # Keep notebook module’s globals aligned with train-time meta
    nbmod.token_maps = meta.get("token_maps", {})
    nbmod.column_order = meta.get("column_order", list(meta.get("categorical_columns", [])) + list(meta.get("numerical_columns", [])))
    nbmod.categorical_columns = meta.get("categorical_columns", [])
    nbmod.numerical_columns = meta.get("numerical_columns", [])


def build_token_constraints(tokenizer, meta):
    """Allow only valid categorical value tokens immediately after each <CAT_col> marker."""
    token_constraints = {}
    tmap = meta.get("token_maps", {})
    for col, vmap in tmap.items():
        key = f"<CAT_{col}>"
        allowed_ids = []
        for tok in vmap.values():
            tid = tokenizer.token2id.get(tok)
            if tid is not None:
                allowed_ids.append(tid)
        token_constraints[key] = allowed_ids
    return token_constraints


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_dir", type=Path, default=Path("models"), help="Directory with dataset subfolders")
    ap.add_argument("--dataset_name", required=True, help="e.g. home_credit | paysim")
    ap.add_argument("--nb_path", type=Path, required=True, help="Training notebook that defines GPTMini/SimpleTokenizer/generate_synthetic_row")
    ap.add_argument("--input_csv", type=Path, required=True, help="CSV with rows to impute (NaNs will be filled)")
    ap.add_argument("--out", type=Path, required=True, help="Where to write the imputed CSV")
    ap.add_argument("--max_rows", type=int, default=0, help="Cap number of rows to process (0 = all)")
    args = ap.parse_args()

    model_dir = args.models_dir / args.dataset_name
    tok_path, mdl_path, meta_path = resolve_artifact_paths(model_dir, args.dataset_name)
    if not tok_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tok_path}")
    if mdl_path is None:
        raise FileNotFoundError(f"Model weights not found in {model_dir} (looked for *transformer_model.pt / *transformer.pt)")
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta not found: {meta_path}")

    # Load meta (schema + token maps + numeric encoding)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    # Import notebook (seed with schema so cells referencing globals don't fail)
    nbmod = load_notebook_module(args.nb_path, schema=meta)

    # Sync notebook globals with the meta.json (token_maps, columns, etc.)
    sync_globals_from_meta(nbmod, meta)

    # Load tokenizer and model
    tokenizer = _load_tokenizer(tok_path, nbmod.SimpleTokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nbmod.GPTMini(vocab_size=len(tokenizer.token2id)).to(device)
    state = torch.load(mdl_path, map_location="cpu")  # state_dict file
    model.load_state_dict(state)
    model.eval()

    # Build categorical constraints
    token_constraints = build_token_constraints(tokenizer, meta)

    # Load input rows
    df_in = pd.read_csv(args.input_csv)
    if args.max_rows and len(df_in) > args.max_rows:
        df_in = df_in.head(args.max_rows)
        print(f"⚠️ Restricted to first {args.max_rows} rows for speed.")

    # Ensure all schema columns exist; extra columns are preserved, but only schema columns are considered for imputation
    schema_cols = meta.get("columns", meta.get("column_order", []))
    for c in schema_cols:
        if c not in df_in.columns:
            df_in[c] = pd.NA

    # Impute only missing values per row
    rows_out = []
    for _, row in tqdm(df_in.iterrows(), total=len(df_in), desc="Imputing"):
        if row.isna().any():
            gen = nbmod.generate_synthetic_row(
                model=model,
                tokenizer=tokenizer,
                device=device,
                token_constraints=token_constraints,
                max_len=1024
            )
            if gen:
                # fill NaNs only for schema columns that the generator produced
                for col in schema_cols:
                    if pd.isna(row.get(col, pd.NA)) and col in gen:
                        row[col] = gen[col]
        rows_out.append(row)

    df_out = pd.DataFrame(rows_out)

    # Keep schema-defined order first, then any extra columns
    ordered = [c for c in schema_cols if c in df_out.columns]
    extras = [c for c in df_out.columns if c not in ordered]
    df_out = df_out[ordered + extras]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.out, index=False)
    print(f"✅ Saved imputed rows: {args.out} | rows={len(df_out)}")


if __name__ == "__main__":
    main()
