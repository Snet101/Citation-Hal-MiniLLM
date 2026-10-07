# Citation Hallucination Audit

A reproducible pipeline for auditing citation hallucinations in **TinyLlama-1.1B-Chat** using regex-based citation detection, sentence embeddings, FAISS retrieval, and post-hoc retrieval-augmented generation (RAG).

This project evaluates whether citation-like language produced by a small open-source language model is actually grounded in retrievable evidence.

## Overview

The experiment evaluates **250 harmful prompts from WildJailbreak**, each under two conditions:

1. **Neutral** — the original prompt
2. **With Citation** — the same prompt with `"Please cite your sources."` appended

This produces **500 total model responses**.

Responses are generated locally with **TinyLlama-1.1B-Chat** using deterministic decoding.

Citation-like sentences are then extracted and verified against a **10,000-document FineWeb-Edu retrieval corpus** using:

- `all-MiniLM-L6-v2` sentence embeddings
- FAISS similarity search
- cosine similarity threshold of **0.60**

Responses without citation-like text, or whose citation-like statements do not meet the grounding threshold, are treated as unsupported under the strict evaluation protocol.

## Pipeline

```text
WildJailbreak prompts
        ↓
build_manifest.py
        ↓
250 prompts × 2 variants
        ↓
run_inference.py
        ↓
TinyLlama responses
        ↓
extractcitations.py
        ↓
citation-like statements
        ↓
verify_citations_fineweb.py
        ↓
MiniLM embeddings + FAISS retrieval
        ↓
supported / unsupported
        ↓
select_unsupported.py
        ↓
retrieve_context.py
        ↓
rag_correct.py
        ↓
retrieval-augmented regeneration
        ↓
reverify_corrected.py
        ↓
final_reverify.py
        ↓
final hallucination metrics
```

## Results

Strict retrieval verification found very high citation hallucination rates:

| Prompt Condition | Supported | Total | Unsupported | Hallucination Rate |
|---|---:|---:|---:|---:|
| Neutral | 6 | 250 | 244 | **97.6%** |
| With Citation | 4 | 250 | 246 | **98.4%** |

Requesting citations did not improve grounding. Under this evaluation, the citation-requested condition performed slightly worse.

A post-hoc RAG correction was also tested. Unsupported responses were paired with retrieved evidence, regenerated using an instruction to rely only on those sources, and then evaluated again using the same verification procedure.

The correction produced **less than a 1 percentage-point improvement** in strict hallucination rate.

## Repository Structure

```text
citation-hallucination-audit/
│
├── src/
│   ├── build_manifest.py
│   ├── run_inference.py
│   ├── extractcitations.py
│   ├── verify_citations_fineweb.py
│   ├── select_unsupported.py
│   ├── retrieve_context.py
│   ├── rag_correct.py
│   ├── reverify_corrected.py
│   └── final_reverify.py
│
├── analysis/
│   ├── bootstrap_metrics.py
│   ├── summarize_citation_rates.py
│   └── plot_hallucination_rates.py
│
├── artifacts/
│   ├── prompts_manifest.csv
│   ├── tinyllama_completions.jsonl
│   ├── verification_results.jsonl
│   ├── hal_summary_fineweb_FINAL.csv
│   ├── unsupported.jsonl
│   ├── unsupported_with_context.jsonl
│   └── corrected_completions.jsonl
│
└── citation_hallucination_audit.pdf
```

## Core Components

### Prompt Manifest Generation

`build_manifest.py`

Samples 250 harmful prompts from WildJailbreak and creates two versions of each:

- neutral
- citation-requested

A fixed random seed is used for reproducibility.

### Model Inference

`run_inference.py`

Runs local text generation with:

- **Model:** TinyLlama/TinyLlama-1.1B-Chat-v1.0
- deterministic decoding
- `do_sample=False`
- `max_new_tokens=128`

Generated responses and prompt metadata are stored as JSONL records.

### Citation Extraction

`extractcitations.py`

Uses regular expressions to identify citation-like language including:

- attribution phrases
- author/year references
- numbered citations
- URLs
- source statements

### Retrieval-Based Verification

`verify_citations_fineweb.py`

Citation-like sentences are embedded using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The embeddings are compared against a FAISS index constructed from a FineWeb-Edu corpus.

A statement is considered supported when its best retrieved match reaches:

```text
cosine similarity >= 0.60
```

The script produces per-response verification logs containing support labels and similarity scores.

### Retrieval-Augmented Correction

The mitigation pipeline consists of:

```text
select_unsupported.py
        ↓
retrieve_context.py
        ↓
rag_correct.py
        ↓
reverify_corrected.py
```

Unsupported responses are selected, relevant passages are retrieved through FAISS, and TinyLlama regenerates answers using instructions to rely only on the provided evidence.

The corrected responses are then evaluated using the same grounding procedure.

### Statistical Analysis

`bootstrap_metrics.py`

Computes bootstrap confidence intervals for hallucination rates.

`plot_hallucination_rates.py`

Visualizes hallucination rates across prompting conditions.

`summarize_citation_rates.py`

Aggregates citation-generation rates across experiment conditions.

## Tech Stack

- Python
- Hugging Face Transformers
- TinyLlama
- SentenceTransformers
- FAISS
- pandas
- NumPy
- NLTK
- matplotlib
- seaborn
- tqdm

## Reproducibility

The repository includes the experiment manifest, model outputs, verification logs, intermediate RAG artifacts, and final summary statistics.

The FineWeb-Edu retrieval index itself is not included because of its size. The experiment used a **10,000-document FineWeb-Edu slice** embedded with `all-MiniLM-L6-v2`.

## Limitations

This is a focused audit rather than a general benchmark.

- Only **TinyLlama-1.1B-Chat** was evaluated.
- The final experiment uses the **vanilla-harmful** WildJailbreak subset.
- FineWeb-Edu provides only partial coverage of possible source material.
- A cosine-similarity threshold is a retrieval-based proxy for grounding rather than definitive factual verification.
- Responses without detectable citation-like text are treated as unsupported under the strict evaluation protocol.
- Results should not be generalized to larger or more capable language models.

## Paper

The full methodology, results, limitations, and discussion are available in:

[`citation_hallucination_audit.pdf`](./citation_hallucination_audit.pdf)

## Authors

**Sanvi Nethikunta**  
**Rithik Gumpu**
