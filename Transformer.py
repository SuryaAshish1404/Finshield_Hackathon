import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pickle
from tqdm.auto import tqdm

column_order = ['alice', 'david', 'emily', 'jacob', 'james', 'john', 'mike', 'lucas', 'mary', 'sarah']
categorical_columns = ['alice', 'emily', 'john']
numerical_columns = [c for c in column_order if c not in categorical_columns]
token_maps = {}

df = pd.read_csv("sequential_training.csv")

token_counter = 0
for col in categorical_columns:
    unique_vals = df[col].unique()
    mapping = {val: f"T{token_counter + i}" for i, val in enumerate(unique_vals)}
    token_maps[col] = mapping
    token_counter += len(mapping)

def number_to_tokens_fixed(value):
    if pd.isna(value): return ['P', 'NAN', 'NAN', 'NAN']
    sign = 'P' if value >= 0 else 'N'
    scaled = int(round(abs(value) * 10))
    base = 16
    digits = []
    for _ in range(3):
        digits.insert(0, str(scaled % base))
        scaled //= base
    return [sign] + digits

def tokenize_row(row):
    tokens = []
    for col in column_order:
        if col in categorical_columns:
            tokens.append(f"<CAT_{col}>")
            tokens.append(token_maps[col].get(row[col], "T_NAN"))
        else:
            tokens.append(f"<NUM_{col}>")
            tokens.extend(number_to_tokens_fixed(row[col]))
    tokens.append("<EOR>")
    return tokens

class SimpleTokenizer:
    def __init__(self):
        self.token2id = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2}
        self.id2token = {0: "<PAD>", 1: "<BOS>", 2: "<EOS>"}
        self.next_id = 3

    def fit(self, sequences):
        for seq in sequences:
            for token in seq:
                if token not in self.token2id:
                    self.token2id[token] = self.next_id
                    self.id2token[self.next_id] = token
                    self.next_id += 1

    def encode(self, seq):
        return [self.token2id["<BOS>"]] + [self.token2id[t] for t in seq] + [self.token2id["<EOS>"]]

    def decode(self, ids):
        return [self.id2token[i] for i in ids if i not in (0, 1, 2)]

class TabularDataset(Dataset):
    def __init__(self, sequences, tokenizer):
        self.data = [tokenizer.encode(seq) for seq in sequences]

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx][:-1])
        y = torch.tensor(self.data[idx][1:])
        return x, y

    def __len__(self):
        return len(self.data)

# --- Model ---
class CustomDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, tgt, tgt_mask=None):
        # Self-attention block
        attn_output, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt2 = self.norm1(tgt + self.dropout1(attn_output))

        # Feed-forward block
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(tgt2))))
        tgt3 = self.norm2(tgt2 + self.dropout2(ff_output))

        return tgt3

class GPTMini(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=8, num_layers=4, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.zeros(1, 512, d_model))
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            CustomDecoderLayer(d_model, nhead, dropout)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embed(x) + self.pos[:, :x.size(1)]
        x = self.dropout(x)
        x = x.transpose(0, 1)  # seq_len, batch, d_model

        mask = nn.Transformer.generate_square_subsequent_mask(x.size(0)).to(x.device)

        for layer in self.layers:
            x = layer(x, tgt_mask=mask)

        x = x.transpose(0, 1)  # batch, seq_len, d_model
        return self.fc(x)


import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

def generate_synthetic_row(model, tokenizer, device, token_constraints, max_len=256):
    model.eval()
    input_ids = [tokenizer.token2id["<BOS>"]]
    decoded_tokens = []
    expecting_num = False
    num_step = 0

    for _ in range(max_len):
        x = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
        logits = logits[0, -1, :]

        if decoded_tokens:
            last_token = decoded_tokens[-1]

            if last_token.startswith("<CAT_"):
                allowed_ids = token_constraints.get(last_token, [])
                if not allowed_ids:
                    return None
                masked_logits = torch.full_like(logits, float('-inf'))
                masked_logits[allowed_ids] = logits[allowed_ids]
                logits = masked_logits

            elif last_token.startswith("<NUM_"):
                expecting_num = True
                num_step = 0

            elif expecting_num:
                all_ids = set(range(len(tokenizer.token2id)))
                disallowed_ids = set()
                for cat_key in token_constraints:
                    disallowed_ids.update(token_constraints[cat_key])
                allowed_ids = list(all_ids - disallowed_ids)

                masked_logits = torch.full_like(logits, float('-inf'))
                masked_logits[allowed_ids] = logits[allowed_ids]
                logits = masked_logits

                num_step += 1
                if num_step == 4:
                    expecting_num = False

        probs = F.softmax(logits, dim=-1)
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            return None

        next_token = torch.multinomial(probs, num_samples=1).item()
        token_str = tokenizer.id2token.get(next_token, "")

        if token_str in ["<EOS>", "<EOR>"]:
            break

        input_ids.append(next_token)
        decoded_tokens.append(token_str)

    # print("Generated tokens:", decoded_tokens)

    row = {}
    i = 0
    # print(i)
    while i < len(decoded_tokens):
        # print(i, decoded_tokens[i])
        if decoded_tokens[i] == "<EOR>":
            break
        token = decoded_tokens[i]
        if token.startswith("<CAT_"):
            current_col = token[5:-1]
            i += 1
            row[current_col] = next((k for k, v in token_maps[current_col].items() if v == decoded_tokens[i]), None)

            # print(f"Set {current_col} to {row[current_col]}")
        elif token.startswith("<NUM_"):
            current_col = token[5:-1]
            i += 1
            if i + 3 >= len(decoded_tokens):
                # print(f"Not enough tokens for {current_col}, expected 4 but got {len(decoded_tokens) - i}")
                return None
            try:
                sign = 1 if decoded_tokens[i] == 'P' else -1
                digits = [int(decoded_tokens[i + j]) for j in range(1, 4)]
                value = sign * (digits[0]*16**2 + digits[1]*16 + digits[2]) / 10.0
                row[current_col] = value
                # print(f"Set {current_col} to {value}")
                if value != value:  # Check for NaN
                    row[current_col] = float("nan")
            except Exception:
                # print(f"Error processing numerical value for {current_col}: {decoded_tokens[i:i+4]}")
                row[current_col] = float("nan")
            i += 3
        i += 1
    # print("Final row:", row)
    return row 

def decode_generated_tokens(tokens):
    row = {}
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.startswith("<CAT_"):
            col = token[5:-1] if token.endswith(">") else token[5:]
            i += 1
            row[col] = next((k for k, v in token_maps[col].items() if v == tokens[i]), None)
        elif token.startswith("<NUM_"):
            col = token[5:-1] if token.endswith(">") else token[5:]
            i += 1
            if tokens[i] == 'NAN' or tokens[i+1] == 'NAN':
                row[col] = float("nan")
            else:
                sign = 1 if tokens[i] == 'P' else -1
                digits = [int(tokens[i + j]) for j in range(1, 4)]
                val = sign * (digits[0]*16**2 + digits[1]*16 + digits[2]) / 10.0
                row[col] = val
            i += 3
        else:
            i += 1
    return row if len(row) == len(column_order) else None

if __name__ == "__main__":
    tokenized = df.apply(tokenize_row, axis=1).tolist()
    tokenizer = SimpleTokenizer()
    tokenizer.fit(tokenized)

    dataset = TabularDataset(tokenized, tokenizer)
    # print(dataset[0])  # Check the first item
    train_len = int(0.8 * len(dataset))
    train_data, val_data = random_split(dataset, [train_len, len(dataset) - train_len])
    pad = lambda x: tuple(nn.utils.rnn.pad_sequence(t, batch_first=True) for t in zip(*x))
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True, collate_fn=pad)
    val_loader = DataLoader(val_data, batch_size=8, collate_fn=pad)

    model = GPTMini(len(tokenizer.token2id)).to("cuda" if torch.cuda.is_available() else "cpu")
    # print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(30):
        model.train()
        total = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out.view(-1, out.size(-1)), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"Epoch {epoch+1} Train Loss: {total / len(train_loader):.4f}")

        model.eval()
        vtotal = 0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = loss_fn(out.view(-1, out.size(-1)), y.view(-1))
                vtotal += loss.item()
        print(f"Epoch {epoch+1} Val Loss: {vtotal / len(val_loader):.4f}")

    torch.save(model.state_dict(), "transformer_model.pt")
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

__all__ = ["GPTMini", "SimpleTokenizer", "token_maps", "column_order", "generate_synthetic_row"]
