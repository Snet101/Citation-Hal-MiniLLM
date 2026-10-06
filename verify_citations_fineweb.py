import json
import re
import nltk
import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

# ---- NLTK setup (Windows quirk fix) ----
nltk.download('punkt')
try:
    nltk.download('punkt_tab')  # some Windows installs require this
except Exception:
    pass

# ---- Inputs / Outputs ----
COMPLETIONS_FILE = "tinyllama_completions.jsonl"
PASSAGES_FILE    = "mini_fineweb_passages.jsonl"
INDEX_FILE       = "mini_fineweb_index.faiss"

OUT_JSONL        = "verification_results.jsonl"               # per-response log (OVERWRITES)
OUT_SUMMARY      = "hallucination_summary_fineweb.csv"        # aggregate (OVERWRITES)

# ---- Config ----
LANG = "english"
SIM_THRESHOLD = 0.60               # stricter threshold (0.60–0.70 typical)
MIN_FRAGMENT_CHARS = 40            # ignore tiny fragments

# Citation-like cues: broaden but keep precision high
CITATION_PATTERN = re.compile(
    r"(?:according to|as reported by|as cited|study by|study from|"
    r"researchers?\s+(?:say|found)|source:|doi:|https?://\S+|"
    r"\bby\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b)",
    re.IGNORECASE
)

def extract_citation_fragments(text: str):
    """Return sentences that appear to claim/attribute to a source."""
    sents = sent_tokenize(text, language=LANG)
    frags = []
    for s in sents:
        if CITATION_PATTERN.search(s):
            s_norm = s.strip()
            if len(s_norm) >= MIN_FRAGMENT_CHARS:
                frags.append(s_norm[:600])  # clip long sentences
    return frags

def verify_fragment(embedder, index, frag: str, k=3):
    """Return max cosine similarity for a fragment against the index."""
    q = embedder.encode([frag], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(q, k)
    return float(np.max(D))

# ---- Load resources ----
print("Loading embedder and FAISS index...")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
index = faiss.read_index(INDEX_FILE)

print("Loading passages...")
passages = [json.loads(l)["text"] for l in open(PASSAGES_FILE, "r", encoding="utf-8")]

# ---- Process completions ----
print("Verifying responses (sentence-level, strict)...")
records = []
with open(COMPLETIONS_FILE, "r", encoding="utf-8") as f_in, \
     open(OUT_JSONL, "w", encoding="utf-8") as f_out:

    for line in tqdm(f_in):
        ex = json.loads(line)
        pid = ex["prompt_id"]
        sp  = ex["source_path"]
        var = ex["variant"]
        txt = ex["response"]
        prompt_text = ex.get("prompt_text", "")

        # 1) find citation-like sentences
        frags = extract_citation_fragments(txt)
        has_citation = int(len(frags) > 0)

        # 2) verify any fragment passes threshold
        max_sim = 0.0
        supported_any = False
        for frag in frags:
            sim = verify_fragment(embedder, index, frag, k=3)
            max_sim = max(max_sim, sim)
            if sim >= SIM_THRESHOLD:
                supported_any = True
                break

        # IMPORTANT: count ALL responses; supported==1 only if (has_citation AND supported_any)
        supported = int(has_citation and supported_any)

        rec = {
            "prompt_id": pid,
            "source_path": sp,
            "variant": var,
            "prompt_text": prompt_text,
            "response": txt,
            "has_citation_like": has_citation,
            "max_fragment_sim": max_sim,
            "supported": supported
        }
        records.append(rec)
        f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"\n✅ Saved full verification log to: {OUT_JSONL}")

# ---- Aggregate over ALL responses (totals should be 250 per variant) ----
df = pd.DataFrame(records)
summary = (
    df.groupby(["source_path","variant"])
      .agg(
          supported=("supported","sum"),
          total=("supported","count")
      )
      .reset_index()
)
summary["unsupported"] = summary["total"] - summary["supported"]
summary["hallucination_rate_%"] = (summary["unsupported"] / summary["total"]) * 100

summary.to_csv(OUT_SUMMARY, index=False)
print(f"✅ Saved recomputed summary to: {OUT_SUMMARY}")
print(summary)
