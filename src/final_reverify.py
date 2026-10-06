import pandas as pd

# Inputs
BASE_FILE = "verification_results.jsonl"
CORR_FILE = "corrected_completions.jsonl"
OUT_CSV   = "hallucination_summary_fineweb_FINAL.csv"

# Load original strict eval results
before = pd.read_json(BASE_FILE, lines=True)

# Load only fields needed from corrected
corr = pd.read_json(CORR_FILE, lines=True)[["prompt_id", "variant", "supported"]]
corr = corr.rename(columns={"supported": "supported_new"})

# Merge corrected support labels
after = before.merge(
    corr,
    on=["prompt_id", "variant"],
    how="left"
)

# Replace supported where updated
after["supported"] = after["supported_new"].fillna(after["supported"])
after.drop(columns=["supported_new"], inplace=True)

# Recompute hallucination stats
summary = (
    after.groupby(["source_path","variant"])
    .agg(
        supported=("supported", "sum"),
        total=("supported", "count")
    )
    .reset_index()
)
summary["unsupported"] = summary["total"] - summary["supported"]
summary["hallucination_rate_%"] = (
    summary["unsupported"] / summary["total"] * 100
)

summary.to_csv(OUT_CSV, index=False)
print("✅ Saved final summary →", OUT_CSV)
print(summary)
