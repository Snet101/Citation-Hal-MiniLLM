import json
import pandas as pd

INFILE = "verification_results.jsonl"
OUTFILE = "unsupported.jsonl"

records = []
with open(INFILE, "r", encoding="utf-8") as f:
    for line in f:
        ex = json.loads(line)
        if ex["has_citation_like"] == 1 and ex["supported"] == 0:
            records.append(ex)

with open(OUTFILE, "w", encoding="utf-8") as f_out:
    for r in records:
        f_out.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"✅ Selected {len(records)} unsupported responses → {OUTFILE}")
