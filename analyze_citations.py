import pandas as pd

df = pd.read_csv("extracted_citations.csv")

summary = (
    df.groupby(["source_path", "variant"])
      .agg(
          responses=("prompt_id", "count"),
          cited=("num_citations", lambda x: (x > 0).sum()),
      )
      .reset_index()
)

summary["citation_rate_%"] = (summary["cited"] / summary["responses"]) * 100

print(summary)
summary.to_csv("citation_summary.csv", index=False)
print("\n✅ Summary saved to citation_summary.csv")
