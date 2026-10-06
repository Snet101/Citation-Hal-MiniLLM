import json
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

INFILE = "unsupported.jsonl"
PASSAGES_FILE = "mini_fineweb_passages.jsonl"
INDEX_FILE = "mini_fineweb_index.faiss"
OUTFILE = "unsupported_with_context.jsonl"

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
index = faiss.read_index(INDEX_FILE)

passages = [json.loads(l)["text"] for l in open(PASSAGES_FILE, "r", encoding="utf-8")]

def retrieve_context(text):
    q = embedder.encode([text], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(q, k=3)
    ctx = [passages[idx] for idx in I[0]]
    return ctx

records = []
with open(INFILE, "r", encoding="utf-8") as f_in:
    for line in tqdm(f_in):
        ex = json.loads(line)
        best_context = retrieve_context(ex["response"])
        ex["retrieved_passages"] = best_context
        records.append(ex)

with open(OUTFILE, "w", encoding="utf-8") as f_out:
    for r in records:
        f_out.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"✅ Retrieved supporting passages → {OUTFILE}")
