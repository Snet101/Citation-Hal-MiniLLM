import pandas as pd
import random
import csv

# ---------------- CONFIG ----------------
SEED = 42
random.seed(SEED)
INPUT_FILE = "train.tsv"           # your dataset file
OUT_PATH = "prompts_manifest.csv"  # output manifest
PER_PATH = 250                     # number of samples per data_type
# ----------------------------------------

# ---- READ TSV DIRECTLY ----
df = pd.read_csv(INPUT_FILE, sep="\t", dtype=str, keep_default_na=False)
print("Columns:", df.columns)

# ---- SAMPLE VANILLA-HARMFUL PROMPTS ----
SOURCE_PATH = "vanilla_harmful"

print(f"Filtering {SOURCE_PATH}...")

subset = df[df["data_type"] == SOURCE_PATH]
texts = subset["vanilla"].tolist()

random.shuffle(texts)
sampled = texts[:PER_PATH]

rows = []

for pid, text in enumerate(sampled):
    rows.append({
        "prompt_id": pid,
        "prompt_text": text,
        "source_path": SOURCE_PATH
    })

# ---- CREATE VARIANTS ----
manifest = []
for r in rows:
    # Neutral version
    manifest.append({**r, "variant": "neutral"})
    # "Please cite" version
    manifest.append({
        **r,
        "variant": "with_citation",
        "prompt_text": r["prompt_text"] + " Please cite your sources."
    })

# ---- SAVE CSV ----
with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["prompt_id", "variant", "prompt_text", "source_path"]
    )
    writer.writeheader()
    writer.writerows(manifest)

print(f"\n🎉 Done! Wrote {OUT_PATH} with {len(manifest)} rows (seed={SEED})")
