import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math
import json

# ==== CONFIG ====
dataset = "home_credit"   # or "paysim"
base_dir = f"data/processed/{dataset}"

real_path = f"{base_dir}/train.csv"
synthetic_path = f"{base_dir}/synthetic_output.csv"
schema_path = f"{base_dir}/schema.json"
plots_dir = f"plots/{dataset}"

# ==== LOAD DATA ====
real_df = pd.read_csv(real_path)
synthetic_df = pd.read_csv(synthetic_path)

# Load schema for col types
with open(schema_path, "r") as f:
    schema = json.load(f)
categorical_cols = schema["categoricals"]
numerical_cols = schema["numericals"]

# Create plots folder
os.makedirs(plots_dir, exist_ok=True)

# ==== PLOTS ====
def plot_numeric_distributions(real_df, synthetic_df, columns, folder=plots_dir):
    for col in columns:
        if col not in real_df or col not in synthetic_df:
            continue
        plt.figure(figsize=(6, 4))
        sns.kdeplot(real_df[col].dropna(), label="Real", fill=True)
        sns.kdeplot(synthetic_df[col].dropna(), label="Synthetic", fill=True, linestyle="--")
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{folder}/{col}_distribution.png")
        plt.close()

def plot_categorical_distributions(real_df, synthetic_df, columns, folder=plots_dir):
    for col in columns:
        if col not in real_df or col not in synthetic_df:
            continue
        real_counts = real_df[col].value_counts(normalize=True)
        synthetic_counts = synthetic_df[col].value_counts(normalize=True)
        
        combined = pd.DataFrame({'Real': real_counts, 'Synthetic': synthetic_counts}).fillna(0)
        ax = combined.plot(kind='bar', stacked=False, figsize=(8, 4))
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Proportion")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{folder}/{col}_categorical_distribution.png")
        plt.close()

def compare_distributions_grid(real_df, synthetic_df, columns, ncols=3):
    nrows = math.ceil(len(columns) / ncols)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*5, nrows*4))

    for i, col in enumerate(columns):
        if col not in real_df or col not in synthetic_df:
            continue
        r, c = divmod(i, ncols)
        ax = axs[r, c] if nrows > 1 else axs[c]
        sns.kdeplot(real_df[col].dropna(), label="Real", fill=True, ax=ax)
        sns.kdeplot(synthetic_df[col].dropna(), label="Synthetic", fill=True, linestyle="--", ax=ax)
        ax.set_title(f"{col}")
        ax.legend()

    for j in range(i+1, nrows*ncols):
        r, c = divmod(j, ncols)
        axs[r, c].axis("off") if nrows > 1 else axs[c].axis("off")

    plt.tight_layout()
    plt.show()

# ==== SUMMARY STATS ====
def calculate_statistics(df, columns):
    stats = {}
    for col in columns:
        if col not in df:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max()
            }
        else:
            stats[col] = df[col].value_counts(normalize=True).to_dict()
    return stats

real_stats = calculate_statistics(real_df, real_df.columns)
synthetic_stats = calculate_statistics(synthetic_df, synthetic_df.columns)

# ==== RUN ====
plot_numeric_distributions(real_df, synthetic_df, numerical_cols)
plot_categorical_distributions(real_df, synthetic_df, categorical_cols)
# Example showcase grid (choose a few numerical cols to compare together)
compare_distributions_grid(real_df, synthetic_df, numerical_cols[:6])
