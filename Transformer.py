# Transformer.py — CSV-only, schema-driven, fixed-length numeric tokens (sign + 10 hex digits)
# Reads:  <data_dir>/{train.csv, schema.json}
# Saves:  <models_dir>/{transformer_model.pt, tokenizer.pkl, meta.json}
# Exports: GPTMini, SimpleTokenizer, token_maps, column_order, categorical_columns, numerical_columns, generate_synthetic_row

import argparse, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm.auto import tqdm

# =========================
# Numeric encoding (FIXED LENGTH)
# =========================
BASE = 16          # hexadecimal "digits" 0..15 as tokens
SCALE = 100        # 2-decimal precision (value ≈ int_tokens / 100)
NUM_DIGITS = 10    # <-- fixed number of base-16 digits per numeric
TOKENS_PER_NUMBER = 1 + NUM_DIGITS   # 1 sign + NUM_DIGITS digits

# =========================
# Globals (also written to meta.json)
# =========================
token_maps = {}            # {col: {value_str -> token_symbol}}
column_order = []          # categoricals + numericals
categorical_columns = []   # list[str]
numerical_columns = []     # list[str]

# =========================
# Tokenization utilities
# =========================
def number_to_tokens_fixed(value: float):
    """
    Encode a numeric value as [sign] + [d_(NUM_DIGITS-1), ..., d_0].
    Missing -> ['P'] + ['NAN'] * NUM_DIGITS  (keeps length constant).
    """
    if pd.isna(value):
        return ['P'] + ['NAN'] * NUM_DIGITS
    sign = 'P' if value >= 0 else 'N'
    scaled = int(round(abs(float(value)) * SCALE))
    digits = []
    for _ in range(NUM_DIGITS):
        digits.insert(0, str(scaled % BASE))
        scaled //= BASE
    return [sign] + digits

def build_token_maps(train_df: pd.DataFrame, categoricals):
    maps = {}
    tok_id = 0
    for col in categoricals:
        uniq = train_df[col].dropna().astype(str).unique()
        mapping = {val: f"T{tok_id + i}" for i, val in enumerate(uniq)}
        mapping["__UNK__"] = f"T{tok_id + len(mapping)}"
        maps[col] = mapping
        tok_id += len(mapping) + 1
    return maps

def tokenize_row(row, column_order, categorical_columns):
    toks = []
    for col in column_order:
        if col in categorical_columns:
            toks.append(f"<CAT_{col}>")
            key = str(row[col]) if not pd.isna(row[col]) else "__UNK__"
            toks.append(token_maps[col].get(key, token_maps[col]["__UNK__"]))
        else:
            toks.append(f"<NUM_{col}>")
            toks.extend(number_to_tokens_fixed(row[col]))
    toks.append("<EOR>")
    return toks

class SimpleTokenizer:
    def __init__(self):
        self.token2id = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2}
        self.id2token = {0: "<PAD>", 1: "<BOS>", 2: "<EOS>"}
        self.next_id = 3
    def fit(self, sequences):
        for seq in sequences:
            for t in seq:
                if t not in self.token2id:
                    self.token2id[t] = self.next_id
                    self.id2token[self.next_id] = t
                    self.next_id += 1
    def encode(self, seq):
        return [self.token2id["<BOS>"]] + [self.token2id[t] for t in seq] + [self.token2id["<EOS>"]]
    def decode(self, ids):
        return [self.id2token[i] for i in ids if i not in (0,1,2)]

class TabularDataset(Dataset):
    def __init__(self, sequences, tokenizer):
        self.data = [torch.tensor(tokenizer.encode(seq), dtype=torch.long) for seq in sequences]
    def __getitem__(self, idx):
        x = self.data[idx][:-1]
        y = self.data[idx][1:]
        return x, y
    def __len__(self):
        return len(self.data)

def pad_batch(batch):
    xs, ys = zip(*batch)
    xpad = nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0)
    ypad = nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=0)
    return xpad, ypad

# =========================
# Model
# =========================
class CustomDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.l1 = nn.Linear(d_model, d_model*4)
        self.l2 = nn.Linear(d_model*4, d_model)
        self.drop = nn.Dropout(dropout)
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)
    def forward(self, tgt, tgt_mask=None):
        attn, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = self.n1(tgt + self.drop(attn))
        ff = self.l2(self.drop(F.relu(self.l1(tgt))))
        tgt = self.n2(tgt + self.drop(ff))
        return tgt

class GPTMini(nn.Module):
    def __init__(self, vocab_size, d_model=192, nhead=8, num_layers=4, dropout=0.2, max_len=1024):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([CustomDecoderLayer(d_model, nhead, dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, vocab_size)
        self.max_len = max_len
    def forward(self, x):
        if x.size(1) > self.max_len:
            raise ValueError(f"Sequence too long ({x.size(1)}) for max_len={self.max_len}")
        h = self.embed(x) + self.pos[:, :x.size(1), :]
        h = self.drop(h).transpose(0,1)  # [L,B,D]
        mask = nn.Transformer.generate_square_subsequent_mask(h.size(0)).to(h.device)
        for layer in self.layers:
            h = layer(h, tgt_mask=mask)
        h = h.transpose(0,1)  # [B,L,D]
        return self.fc(h)

# =========================
# Generation (used by Inference too)
# =========================
def generate_synthetic_row(model, tokenizer, device, token_constraints=None, max_len=1024):
    """
    Sample one row. Enforces:
      - categorical value sets (if constraints provided),
      - exactly TOKENS_PER_NUMBER numeric tokens after a <NUM_col> marker.
    """
    model.eval()
    input_ids = [tokenizer.token2id["<BOS>"]]
    decoded = []

    expecting_num = False
    num_step = 0

    for _ in range(max_len):
        x = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)[0, -1, :]

        last = decoded[-1] if decoded else None
        if last and last.startswith("<CAT_"):
            if token_constraints:
                allowed_ids = token_constraints.get(last, [])
                if allowed_ids:
                    mask = torch.full_like(logits, float("-inf"))
                    mask[allowed_ids] = logits[allowed_ids]
                    logits = mask
        elif last and last.startswith("<NUM_"):
            expecting_num = True
            num_step = 0
        elif expecting_num:
            # Block categorical value tokens during numeric emission
            if token_constraints:
                disallowed = set()
                for ids in token_constraints.values(): disallowed.update(ids)
                all_ids = set(range(len(tokenizer.token2id)))
                allowed = list(all_ids - disallowed)
                mask = torch.full_like(logits, float("-inf"))
                mask[allowed] = logits[allowed]
                logits = mask
            num_step += 1
            if num_step == TOKENS_PER_NUMBER:
                expecting_num = False

        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1).item()
        tok = tokenizer.id2token.get(nxt, "")
        if tok in ["<EOS>", "<EOR>"]:
            break
        input_ids.append(nxt)
        decoded.append(tok)

    # Decode tokens → dict
    row = {}
    i = 0
    while i < len(decoded):
        t = decoded[i]
        if t == "<EOR>":
            break
        if t.startswith("<CAT_"):
            col = t[5:-1]
            i += 1
            rev = {v:k for k, v in token_maps[col].items()}
            val_tok = decoded[i] if i < len(decoded) else None
            row[col] = rev.get(val_tok, None)
        elif t.startswith("<NUM_"):
            col = t[5:-1]
            i += 1
            if i + NUM_DIGITS >= len(decoded):
                return None
            sign_tok = decoded[i]
            digit_toks = decoded[i+1 : i+1+NUM_DIGITS]
            if any(dt == "NAN" for dt in digit_toks):
                val = float("nan")
            else:
                try:
                    acc = 0
                    for d in digit_toks:
                        acc = acc * BASE + int(d)
                    val = (1 if sign_tok == "P" else -1) * (acc / float(SCALE))
                except Exception:
                    val = float("nan")
            row[col] = val
            i += NUM_DIGITS  # consumed sign + digits (sign is at i, then advance over digits)
        i += 1

    # Ensure all columns present
    for c in column_order:
        if c not in row:
            row[c] = (np.nan if c in numerical_columns else None)
    return row

# =========================
# Training entrypoint
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Folder with train.csv and schema.json")
    ap.add_argument("--models_dir", default="models", help="Where to save model/tokenizer/meta")
    ap.add_argument("--dataset_name", default="", help="If set, saves under models_dir/dataset_name/")
    ap.add_argument("--max_rows", type=int, default=50_000, help="Cap rows for speed (0 = no cap)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()


    data_dir = Path(args.data_dir)
    models_dir = Path(args.models_dir)
    if args.dataset_name:
        models_dir = models_dir / args.dataset_name
    models_dir.mkdir(parents=True, exist_ok=True)


    schema = json.loads((data_dir / "schema.json").read_text())
    target = schema["target"]
    cats = list(schema["categoricals"])
    nums = list(schema["numericals"])

    # Keep labels as categoricals in the sequence
    df_full = pd.read_csv(data_dir / "train.csv")
    extra_labels = []
    if "isFraud" in df_full.columns and "isFraud" not in cats:
        extra_labels.append("isFraud")
    if target in df_full.columns and target not in cats:
        extra_labels.append(target)
    cats = cats + [c for c in extra_labels if c not in cats]

    global categorical_columns, numerical_columns, column_order, token_maps
    categorical_columns = cats
    numerical_columns = nums
    column_order = categorical_columns + numerical_columns

    # Optional downsample for speed
    if args.max_rows and len(df_full) > args.max_rows:
        if target in df_full.columns and set(df_full[target].dropna().unique()).issubset({0,1}):
            pos = df_full[df_full[target] == 1]
            neg = df_full[df_full[target] == 0]
            n_pos = min(len(pos), max(1, int(0.1 * args.max_rows)))
            n_neg = min(len(neg), args.max_rows - n_pos)
            df = pd.concat([
                pos.sample(n=n_pos, random_state=42),
                neg.sample(n=n_neg, random_state=42)
            ], axis=0).sample(frac=1.0, random_state=42)
        else:
            df = df_full.sample(n=args.max_rows, random_state=42)
    else:
        df = df_full.copy()

    # Build per-column categorical token maps (train only)
    token_maps = build_token_maps(df, categorical_columns)

    # Tokenize rows
    sequences = [tokenize_row(r, column_order, categorical_columns) for _, r in df.iterrows()]
    tokenizer = SimpleTokenizer()
    tokenizer.fit(sequences)

    # Dataset & loaders
    dataset = TabularDataset(sequences, tokenizer)
    train_len = int(0.9 * len(dataset))
    train_data, val_data = random_split(dataset, [train_len, len(dataset) - train_len])
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=pad_batch)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, collate_fn=pad_batch)

    # Model / optim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTMini(len(tokenizer.token2id)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    # Train
    for epoch in range(args.epochs):
        model.train(); tr_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item()
        tr_loss /= max(1, len(train_loader))

        model.eval(); va_loss = 0.0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                va_loss += loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()
        va_loss /= max(1, len(val_loader))
        print(f"Epoch {epoch+1}: train {tr_loss:.4f} | valid {va_loss:.4f}")

    # Save artifacts
    torch.save(model.state_dict(), models_dir / "transformer_model.pt")
    with open(models_dir / "tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    # Save meta for inference
    meta = {
        "column_order": column_order,
        "categorical_columns": categorical_columns,
        "numerical_columns": numerical_columns,
        "token_maps": token_maps,
        "numeric_encoding": {"base": BASE, "scale": SCALE, "num_digits": NUM_DIGITS, "tokens_per_number": TOKENS_PER_NUMBER}
    }
    (models_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved model/tokenizer/meta to {models_dir}")

if __name__ == "__main__":
    main()

__all__ = [
    "GPTMini",
    "SimpleTokenizer",
    "token_maps",
    "column_order",
    "categorical_columns",
    "numerical_columns",
    "generate_synthetic_row",
]
