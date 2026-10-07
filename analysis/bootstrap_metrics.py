import argparse
import pandas as pd
import numpy as np
from tqdm import trange

def bootstrap_confidence_intervals(df, value_col, group_cols, B=1000, seed=42):
    rng = np.random.default_rng(seed)
    out_rows = []
    for group_vals, sub in df.groupby(group_cols):
        rates = []
        arr = sub[value_col].values
        n = len(arr)
        for _ in trange(B, leave=False, desc=f"Bootstrapping group {group_vals}"):
            sample = rng.choice(arr, n, replace=True)
            rates.append(sample.mean())
        mean = np.mean(rates)
        ci_low, ci_high = np.percentile(rates, [2.5, 97.5])
        out_rows.append({
            **dict(zip(group_cols, group_vals if isinstance(group_vals, tuple) else [group_vals])),
            "mean": mean,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n": n
        })
    return pd.DataFrame(out_rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bootstrap hallucination rates per source/variant")
    
    # VS Code-friendly defaults
    parser.add_argument("--input", dest="infile",
                        default="hallucination_summary_fineweb.csv",
                        help="Input CSV file path (default: hallucination_summary_fineweb.csv)")
    parser.add_argument("--out", dest="outfile",
                        default="bootstrap_summary.json",
                        help="Output JSON file path")
    parser.add_argument("--csv", dest="outcsv",
                        default="bootstrap_summary.csv",
                        help="Output CSV file path")
    parser.add_argument("--B", type=int, default=1000,
                        help="Number of bootstrap samples per group")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()

    # --- Load CSV ---
    try:
        df = pd.read_csv(args.infile)
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file '{args.infile}' not found. Please check the path.")

    # Strip spaces from column names to avoid hidden character issues
    df.columns = df.columns.str.strip()

    # --- Prepare hallucination_rate column ---
    if "hallucination_rate_%" in df.columns:
        df["hallucination_rate"] = df["hallucination_rate_%"] / 100.0
    elif "hallucination_rate" in df.columns:
        pass
    else:
        raise ValueError("Input file must have 'hallucination_rate_%' or 'hallucination_rate' column.")

    # --- Run bootstrap ---
    out_df = bootstrap_confidence_intervals(
        df, 
        "hallucination_rate", 
        ["source_path", "variant"], 
        B=args.B, 
        seed=args.seed
    )

    # --- Save outputs ---
    out_df.to_json(args.outfile, orient="records", indent=2)
    out_df.to_csv(args.outcsv, index=False)

    print(f"\n✅ Bootstrap completed!")
    print(f"Saved JSON: {args.outfile}")
    print(f"Saved CSV:  {args.outcsv}")
    print("\nSample output:")
    print(out_df.head())
