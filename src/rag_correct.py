import json
from transformers import pipeline
from tqdm import tqdm

INFILE = "unsupported_with_context.jsonl"
OUTFILE = "corrected_completions.jsonl"

generator = pipeline("text-generation",
                     model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                     device_map="auto",
                     max_new_tokens=256,
                     do_sample=False)

def build_prompt(prompt_text, passages):
    ctx = "\n\n".join([f"Source {i+1}: {p}" for i, p in enumerate(passages)])
    return (
        "Use ONLY the evidence from the provided sources. "
        "If there is not enough evidence, state 'I cannot verify this.'\n\n"
        f"{ctx}\n\n"
        f"Question: {prompt_text}\nAnswer:"
    )

records = []
with open(INFILE, "r", encoding="utf-8") as f_in, \
     open(OUTFILE, "w", encoding="utf-8") as f_out:

    for line in tqdm(f_in):
        ex = json.loads(line)
        prompt = build_prompt(ex["prompt_text"], ex["retrieved_passages"])
        resp = generator(prompt)[0]["generated_text"]
        ex["corrected_response"] = resp
        records.append(ex)
        f_out.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"✅ Generated corrected responses → {OUTFILE}")
