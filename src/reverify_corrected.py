import json
import re
import nltk
import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

nltk.download('punkt')
try: nltk.download('punkt_tab')
except: pass

INFILE = "corrected_completions.jsonl"
PASSAGES_FILE = "mini_fineweb_passages.jsonl"
INDEX_FILE = "mini_fineweb_index.faiss"
OUTFILE = "hallucination_summary_fineweb_corrected.csv"

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
index = faiss.read_index(INDEX_FILE)
LANG="english"
SIM_THRESHOLD=0.60
MIN_FRAGMENT_CHARS=40

CITATION_PATTERN = re.compile(
    r"(according to|as reported by|as cited|study by|study from|source:|doi:|https?://\S+|"
    r"\bby\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
    re.IGNORECASE
)

def extract_frags(text):
    return [s for s in sent_tokenize(text, language=LANG)
            if CITATION_PATTERN.search(s) and len(s) >= MIN_FRAGMENT_CHARS]

def verify_frag(frag):
    vec = embedder.encode([frag], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(vec, k=3)
    return float(np.max(D)) >= SIM_THRESHOLD

records = []
with open(INFILE, "r", encoding="utf-8") as f:
    for line in tqdm(f):
        ex = json.loads(line)
        frags = extract_frags(ex["corrected_response"])
        supported = any(verify_frag(f) for f in frags)
        records.append({
            "source_path": ex["source_path"],
            "variant": ex["variant"],
            "supported": int(supported)
        })

df = pd.DataFrame(records)
summary = df.groupby(["source_path","variant"]).agg(
    supported=("supported","sum"),
    total=("supported","count")
).reset_index()
summary["unsupported"] = summary["total"] - summary["supported"]
summary["hallucination_rate_%"] = (summary["unsupported"]/summary["total"])*100
summary.to_csv(OUTFILE, index=False)

print(f"✅ Saved corrected summary → {OUTFILE}")
print(summary)
