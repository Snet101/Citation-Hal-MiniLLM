import json
import re
import csv

INPUT_FILE = "tinyllama_completions.jsonl"
OUT_FILE = "extracted_citations.csv"

# Expanded regex patterns
citation_pattern = re.compile(
    r"\((?:[A-Z][a-zA-Z]+,?\s?\d{4})\)|"      # (Smith, 2020)
    r"\[\d+\]|"                               # [1]
    r"https?://\S+|"                          # URLs
    r"(?:Source|By|by|According to)[: ]+[A-Z][^\n]+",  # "By CNN", "Source: Wikipedia"
    re.IGNORECASE
)

def extract_citations(text):
    return citation_pattern.findall(text)

rows = []
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        response = data.get("response", "")
        citations = extract_citations(response)
        rows.append({
            "prompt_id": data["prompt_id"],
            "variant": data["variant"],
            "source_path": data["source_path"],
            "num_citations": len(citations),
            "citations_found": "; ".join(citations),
            "response_preview": response[:200]
        })

with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"✅ Updated citation extraction saved to {OUT_FILE}")
