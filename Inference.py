# Inference.py — fixes KeyError by syncing meta into Transformer module globals
# Loads:  <models_dir>[/<dataset_name>]/{transformer_model.pt, tokenizer.pkl, meta.json}
# Saves:  CSV by default (see --out)

import argparse, json, pickle
from pathlib import Path
import torch
import pandas as pd
from tqdm.auto import tqdm

# Import train-time classes + module to set globals
import Transformer as T
from Transformer import GPTMini, generate_synthetic_row, SimpleTokenizer  # noqa: F401

# ---- Robust tokenizer unpickling (handles __main__.SimpleTokenizer) ----
class _RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__" and name == "SimpleTokenizer":
            from Transformer import SimpleTokenizer as ST
            return ST
        if module == "Transformer" and name == "SimpleTokenizer":
            from Transformer import SimpleTokenizer as ST
            return ST
        return super().find_class(module, name)

def _load_tokenizer(path: Path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except AttributeError:
        with open(path, "rb") as f:
            return _RenameUnpickler(f).load()

def _sync_transformer_globals_from_meta(meta: dict):
    """Populate Transformer module globals so generate_synthetic_row sees them."""
    T.token_maps = meta.get("token_maps", {})
    T.column_order = meta.get("column_order", [])
    T.categorical_columns = meta.get("categorical_columns", [])
    T.numerical_columns = meta.get("numerical_columns", [])

def load_artifacts(models_dir: Path):
    # 1) tokenizer
    tokenizer = _load_tokenizer(models_dir / "tokenizer.pkl")
    # 2) meta (schema + token maps + numeric config)
    meta = json.loads((models_dir / "meta.json").read_text())
    # Push into Transformer globals for generate_synthetic_row()
    _sync_transformer_globals_from_meta(meta)
    # 3) model weights
    model = GPTMini(vocab_size=len(tokenizer.token2id))
    # weights_only=True is safe here since we're loading a state_dict saved by torch.save(model.state_dict())
    state = torch.load(models_dir / "transformer_model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, tokenizer, meta

def build_token_constraints(tokenizer, meta):
    """Allow only valid categorical value tokens right after each <CAT_col> marker."""
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

def postprocess_dataframe(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    # Column order
    cols = meta["column_order"]
    df = df.reindex(columns=cols)

    # Categorical cleanup
    for c in meta["categorical_columns"]:
        if c in df.columns:
            df[c] = df[c].replace({"__UNK__": pd.NA}).astype("string")

    # Numeric cleanup
    for n in meta["numerical_columns"]:
        if n in df.columns:
            df[n] = pd.to_numeric(df[n], errors="coerce")

    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_dir", type=Path, default=Path("models"), help="Folder with model artifacts")
    ap.add_argument("--dataset_name", default="", help="If set, loads from models_dir/dataset_name/")
    ap.add_argument("--n_rows", type=int, default=1000)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--out", type=Path, default=Path("data/processed/synthetic_output.csv"))
    args = ap.parse_args()

    models_path = args.models_dir / args.dataset_name if args.dataset_name else args.models_dir
    model, tokenizer, meta = load_artifacts(models_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    token_constraints = build_token_constraints(tokenizer, meta)

    rows = []
    for _ in tqdm(range(args.n_rows), desc="Generating synthetic rows"):
        r = generate_synthetic_row(
            model=model,
            tokenizer=tokenizer,
            device=device,
            token_constraints=token_constraints,
            max_len=args.max_len
        )
        if r:
            rows.append(r)

    df = pd.DataFrame(rows)
    df = postprocess_dataframe(df, meta)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved synthetic data: {args.out} | rows={len(df)}")

if __name__ == "__main__":
    main()
