# run_inference.py
import json, time, csv, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import pipeline

parser = argparse.ArgumentParser()
parser.add_argument("--manifest", default="prompts_manifest.csv")
parser.add_argument("--out", default="completions.jsonl")
parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
parser.add_argument("--workers", type=int, default=2)   # lower if GPU OOM
parser.add_argument("--max_new_tokens", type=int, default=512)
args = parser.parse_args()

print("Loading generation model:", args.model)
generator = pipeline("text-generation", model=args.model, device_map="auto", torch_dtype="auto")

def call_model(prompt_text):
    start = time.time()
    out = generator(prompt_text, max_new_tokens=args.max_new_tokens, do_sample=False)
    elapsed = time.time() - start
    return {"response": out[0]["generated_text"], "latency_s": elapsed}

# read manifest
rows = []
with open(args.manifest, newline="", encoding="utf-8") as f:
    import csv
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

print(f"Running on {len(rows)} prompts (workers={args.workers})")

with ThreadPoolExecutor(max_workers=args.workers) as ex:
    futures = {ex.submit(call_model, r["prompt_text"]): r for r in rows}
    for fut in as_completed(futures):
        meta = futures[fut]
        out = fut.result()
        out_record = {**meta, **out}
        # append to file
        with open(args.out, "a", encoding="utf-8") as w:
            w.write(json.dumps(out_record, ensure_ascii=False) + "\n")
print("Done.")
