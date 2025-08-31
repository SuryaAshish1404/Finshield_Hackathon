#!/usr/bin/env python3
# Inference.py — import model & helpers from a notebook, load artifacts, generate CSV
# Expects artifacts in: models/<dataset_name>/
#   - <dataset_name>_tokenizer.pkl
#   - <dataset_name>_transformer_model.pt   (falls back to <dataset_name>_transformer.pt)
#   - meta.json
# And schema in: data/processed/<dataset_name>/schema.json

import argparse
import json
import pickle
import types
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm


# -----------------------------
# Load a .ipynb as a Python module, with pre-seeded globals
# -----------------------------
def load_notebook_module(nb_path: Path, module_name: str = "finshield_nb") -> types.ModuleType:
    import nbformat
    from IPython.core.interactiveshell import InteractiveShell

    nb = nbformat.read(str(nb_path), as_version=4)
    module = types.ModuleType(module_name)
    module.__dict__["__file__"] = str(nb_path)

    # PRE-SEED globals to avoid NameError during notebook execution
    exec_env = module.__dict__
    exec_env.setdefault("categorical_columns", [])
    exec_env.setdefault("numerical_columns", [])
    exec_env.setdefault("column_order", [])
    exec_env.setdefault("token_maps", {})

    shell = InteractiveShell.instance()

    for cell in nb.cells:
        if cell.cell_type == "code":
            code = shell.input_transformer_manager.transform_cell(cell.source)
            exec(code, exec_env)

    return module


# -----------------------------
# Robust tokenizer unpickling (map to the notebook's SimpleTokenizer)
# -----------------------------
class _RenameUnpickler(pickle.Unpickler):
    def __init__(self, file, simple_tokenizer_cls):
        super().__init__(file)
        self._st_cls = simple_tokenizer_cls

    def find_class(self, module, name):
        if name == "SimpleTokenizer":
            return self._st_cls
        return super().find_class(module, name)


def _load_tokenizer(path: Path, simple_tokenizer_cls):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except AttributeError:
        with open(path, "rb") as f:
            return _RenameUnpickler(f, simple_tokenizer_cls).load()


# -----------------------------
# Sync schema/meta into the notebook module’s globals
# -----------------------------
def sync_schema_and_meta_into_module(nbmod, schema: dict, meta: dict):
    # Schema-driven globals
    nbmod.column_order = schema.get("columns", [])
    nbmod.categorical_columns = schema.get("categoricals", [])
    nbmod.numerical_columns = schema.get("numericals", [])

    # Meta-driven globals (token maps, numeric encoding info, etc.)
    nbmod.token_maps = meta.get("token_maps", {})

    # If the notebook exposes numeric encoding constants, sync them too (optional)
    ne = meta.get("numeric_encoding", {})
    for k, v in ne.items():
        setattr(nbmod, k, v)


# -----------------------------
# Build categorical constraints (ids allowed after each <CAT_col>)
# -----------------------------
def build_token_constraints(tokenizer, meta):
    token_constraints = {}
    token_maps = meta.get("token_maps", {})
    for col, vmap in token_maps.items():
        key = f"<CAT_{col}>"
        allowed_ids = []
        for tok in vmap.values():
            tid = tokenizer.token2id.get(tok)
            if tid is not None:
                allowed_ids.append(tid)
        token_constraints[key] = allowed_ids
    return token_constraints


def postprocess_dataframe(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    # Ensure all expected columns exist and final CSV order matches schema["columns"]
    expected_cols = list(schema.get("columns", []))
    cats = set(schema.get("categoricals", []))
    nums = set(schema.get("numericals", []))

    # Add missing columns with sensible defaults
    for c in expected_cols:
        if c not in df.columns:
            if c in nums:
                df[c] = pd.Series([pd.NA] * len(df), dtype="float64")
            else:
                df[c] = pd.Series([pd.NA] * len(df), dtype="string")

    # Coerce dtypes / sanitize
    for c in df.columns:
        if c in cats:
            df[c] = df[c].replace({"__UNK__": pd.NA}).astype("string")
        elif c in nums:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Reorder exactly as schema["columns"]
    if expected_cols:
        df = df[expected_cols]

    return df


def resolve_artifact_paths(model_dir: Path, dataset_name: str):
    tok_path = model_dir / f"{dataset_name}_tokenizer.pkl"
    meta_path = model_dir / "meta.json"
    model_candidates = [
        model_dir / f"{dataset_name}_transformer_model.pt",
        model_dir / f"{dataset_name}_transformer.pt",
    ]
    mdl_path = next((p for p in model_candidates if p.exists()), None)
    return tok_path, mdl_path, meta_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_dir", type=Path, default=Path("models"),
                    help="Directory containing dataset subfolders with artifacts")
    ap.add_argument("--dataset_name", required=True,
                    help="Dataset name (e.g., 'home_credit' or 'paysim')")
    ap.add_argument("--nb_path", type=Path, default=Path("HOME-CREDITED-Transformer.ipynb"),
                    help="Path to the training notebook that defines GPTMini/SimpleTokenizer/generate_synthetic_row")
    ap.add_argument("--n_rows", type=int, default=1000)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--out", type=Path, default=Path("data/processed/synthetic_output.csv"))
    args = ap.parse_args()

    # Load schema FIRST so we can seed globals properly
    schema_path = Path("data/processed") / args.dataset_name / "schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema not found: {schema_path}")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    # 1) Import definitions from the notebook with pre-seeded globals
    nbmod = load_notebook_module(args.nb_path, module_name="finshield_nb")

    # Check required exports exist
    required_attrs = ["GPTMini", "SimpleTokenizer", "generate_synthetic_row"]
    missing = [a for a in required_attrs if not hasattr(nbmod, a)]
    if missing:
        raise AttributeError(f"Notebook is missing required definitions: {missing}")

    GPTMini = nbmod.GPTMini
    SimpleTokenizer = nbmod.SimpleTokenizer
    generate_synthetic_row = nbmod.generate_synthetic_row

    # 2) Resolve artifact paths
    model_dir = args.models_dir / args.dataset_name
    tok_path, mdl_path, meta_path = resolve_artifact_paths(model_dir, args.dataset_name)

    # Validate artifacts exist
    if not tok_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tok_path}")
    if mdl_path is None:
        raise FileNotFoundError(
            f"Model weights not found. Looked for:\n"
            f" - {model_dir / (args.dataset_name + '_transformer_model.pt')}\n"
            f" - {model_dir / (args.dataset_name + '_transformer.pt')}"
        )
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta not found: {meta_path}")

    print(f"{args.dataset_name.upper()} — artifacts:")
    print(f"  tokenizer: {tok_path.name}")
    print(f"  model:     {mdl_path.name}")
    print(f"  meta:      {meta_path.name}")
    print(f"  schema:    {schema_path.name}")

    # 3) Load tokenizer & meta
    tokenizer = _load_tokenizer(tok_path, SimpleTokenizer)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    # 4) Sync schema + meta into the notebook module (globals for generation)
    sync_schema_and_meta_into_module(nbmod, schema, meta)

    # 5) Build model & load weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTMini(vocab_size=len(tokenizer.token2id)).to(device)
    state = torch.load(mdl_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    # 6) Build categorical constraints
    token_constraints = build_token_constraints(tokenizer, meta)

    # 7) Generate rows
    rows = []
    for _ in tqdm(range(args.n_rows), desc=f"Generating {args.dataset_name} rows"):
        r = generate_synthetic_row(
            model=model,
            tokenizer=tokenizer,
            device=device,
            token_constraints=token_constraints,
            max_len=args.max_len
        )
        if r:
            rows.append(r)

    # 8) Save
    df = pd.DataFrame(rows)
    df = postprocess_dataframe(df, schema)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved synthetic data: {args.out} | rows={len(df)}")


if __name__ == "__main__":
    main()
