"""Offline evaluator + policy simulator for the dynamic-slice router experiments.

* evaluate(table) replicates RnD/benchmarking_retrievals.ipynb exactly (hybrid search on `vector` + FTS on `text`,
  ColBERT rerank, top-20; Recall / MRR / MAR at 5,10,15,20 with the notebook's formulas).
* simulate(policy) assembles a contextualised corpus from the FIXED-radius JSONs (preprocessed_chunks/
  ablation_doc_slice_radius_{k}.json hold a summary for every chunk at every k) according to a per-chunk k assignment,
  builds a scratch LanceDB table (same schema / embedding function as the notebooks) and evaluates it.  This is a
  proxy for "run the generation notebook with this router": same k -> same prompt to the summariser, up to Ollama's
  temperature-0 jitter.
"""
import json
import os
import sys
import time
from collections import Counter
from typing import Dict, List

import numpy as np
import torch
import lancedb
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from lancedb.rerankers import ColbertReranker

RND = "/home/martin/projects/TDK/Document_Slice_Contextual_Retrieval/RnD"
SCRATCH = os.path.dirname(os.path.abspath(__file__))
SCRATCH_DB = os.path.join(SCRATCH, "simdb")
QNA_FILE_PATH = os.path.join(RND, "q_and_a/Gemini/scientific_multi_chunk_control.json")
EMBEDDING_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"

_hf = None
_reranker = None


def hf():
    global _hf
    if _hf is None:
        _hf = get_registry().get("huggingface").create(name=EMBEDDING_MODEL_NAME, trust_remote_code=True,
                                                       device="cuda" if torch.cuda.is_available() else "cpu")
    return _hf


def reranker():
    global _reranker
    if _reranker is None:
        _reranker = ColbertReranker()
    return _reranker


def qna():
    return json.load(open(QNA_FILE_PATH))


def fixed(k: int) -> Dict[str, dict]:
    recs = json.load(open(os.path.join(RND, f"preprocessed_chunks/ablation_doc_slice_radius_{k}.json")))
    return {r["id"]: r for r in recs}


def dynamic_records() -> List[dict]:
    return json.load(open(os.path.join(RND, "preprocessed_chunks/ablation_doc_slice_radius_dynamic.json")))


# ----------------------------------------------------------------------------------------------------------------------
def make_schema():
    h = hf()

    class SimDocument(LanceModel):
        text: str = h.SourceField()
        vector: Vector(h.ndims()) = h.VectorField()
        original_text: str
        context: str
        document: str
        id: str
        doc_slice_radius: int

    return SimDocument


def build_table(name: str, records: List[dict], db_path: str = SCRATCH_DB):
    """Create (overwrite) a scratch table exactly like the notebooks do: batches of 100, scalar id index, FTS on text."""
    os.makedirs(db_path, exist_ok=True)
    db = lancedb.connect(db_path)
    schema = make_schema()
    db.create_table(name, schema=schema, mode="overwrite")
    table = db.open_table(name)
    rows = [{"text": r["text"], "original_text": r["original_text"], "context": r.get("context", ""), "document": r["document"],
             "id": r["id"], "doc_slice_radius": int(r.get("doc_slice_radius", -1))} for r in records]
    for i in range(0, len(rows), 100):
        table.add(rows[i:i + 100])
    table.create_scalar_index("id", replace=True)
    table.create_fts_index("text", replace=True)
    table.wait_for_index(["text_idx"])
    return table


def open_rnd_table(name: str):
    return lancedb.connect(os.path.join(RND, "db")).open_table(name)


# ----------------------------------------------------------------------------------------------------------------------
def evaluate(table, questions=None, verbose=False):
    """Notebook-identical metrics. Returns dict with 'avg' {N: {recall, mrr, mar}}, 'per_q' list, 'top20' list of id lists."""
    questions = questions or qna()
    rr = reranker()
    per_q, top20 = [], []
    t0 = time.time()
    for qi, question in enumerate(questions):
        supporting = set(question["supporting_chunks"])
        df = table.search(question["question"], query_type="hybrid", vector_column_name="vector", fts_columns="text") \
            .rerank(reranker=rr).limit(20).to_pandas()
        ids = df["id"].tolist()
        top20.append(ids)
        m = {}
        for i in range(5, 21, 5):
            cut = ids[:i]
            recall = sum(1 for c in cut if c in supporting) / len(supporting)
            ranks = [rank + 1 for rank, c in enumerate(cut) if c in supporting]
            m[i] = {"recall": recall,
                    "reciprocal_rank": (1.0 / ranks[0]) if ranks else np.nan,
                    "average_rank": (sum(ranks) / len(ranks)) if ranks else np.nan}
        per_q.append(m)
        if verbose and (qi + 1) % 50 == 0:
            print(f"    {qi + 1}/{len(questions)} questions, {time.time() - t0:.0f}s", flush=True)
    avg = {}
    for i in range(5, 21, 5):
        avg[i] = {"recall": float(np.average([m[i]["recall"] for m in per_q])),
                  "mrr": float(np.nanmean([m[i]["reciprocal_rank"] for m in per_q])),
                  "mar": float(np.nanmean([m[i]["average_rank"] for m in per_q]))}
    return {"avg": avg, "per_q": per_q, "top20": top20}


def fmt(res, label=""):
    a = res["avg"]
    return (f"{label:34s} R@5 {a[5]['recall']:.4f} | R@10 {a[10]['recall']:.4f} | R@15 {a[15]['recall']:.4f} | "
            f"R@20 {a[20]['recall']:.4f}  MRR {a[20]['mrr']:.5f}  MAR {a[20]['mar']:.5f}")


# ----------------------------------------------------------------------------------------------------------------------
_FIXED_CACHE: Dict[int, Dict[str, dict]] = {}


def fixed_cached(k):
    if k not in _FIXED_CACHE:
        _FIXED_CACHE[k] = fixed(k)
    return _FIXED_CACHE[k]


def assemble(policy: Dict[str, int], k0_mode: str = "skip") -> List[dict]:
    """policy: id -> k (0..4). k0_mode: 'skip' (no context, embed the raw chunk) or 'summary' (use the fixed k=0 summary)."""
    base = fixed_cached(2)  # for original_text / document lookup
    out = []
    for cid, k in policy.items():
        if k == 0 and k0_mode == "skip":
            r = base[cid]
            out.append({"text": r["original_text"], "original_text": r["original_text"], "context": "", "document": r["document"],
                        "id": cid, "doc_slice_radius": 0})
        else:
            r = fixed_cached(k)[cid]
            out.append({**r, "doc_slice_radius": k})
    return out


def simulate(name: str, policy: Dict[str, int], k0_mode: str = "skip", questions=None, verbose=False):
    recs = assemble(policy, k0_mode)
    table = build_table(name, recs)
    res = evaluate(table, questions, verbose)
    res["k_dist"] = dict(sorted(Counter(policy.values()).items()))
    return res


def assemble_from_sources(policy: Dict[str, int], sources, k0_mode: str = "skip") -> List[dict]:
    """Like assemble(), but a summary for (chunk, k) is taken from the first of `sources` (dicts id -> record with
    'doc_slice_radius' and 'context') that holds one for exactly that k; the paper's fixed-k JSON is the fallback.
    Lets a policy be evaluated on the summaries a REAL run of this pipeline produced, swapping as few as possible."""
    base = fixed_cached(2)
    out, used = [], Counter()
    for cid, k in policy.items():
        if k == 0 and k0_mode == "skip":
            r = base[cid]
            out.append({"text": r["original_text"], "original_text": r["original_text"], "context": "", "document": r["document"],
                        "id": cid, "doc_slice_radius": 0})
            used["skip"] += 1
            continue
        rec, src_name = None, "fixed_json"
        for name, src in sources:
            r = src.get(cid)
            if r is not None and int(r.get("doc_slice_radius", -1)) == k and (r.get("context") or "").strip():
                rec, src_name = r, name
                break
        if rec is None:
            rec = fixed_cached(k)[cid]
        used[src_name] += 1
        out.append({"text": rec["context"] + "\n\n" + rec["original_text"], "original_text": rec["original_text"], "context": rec["context"],
                    "document": rec["document"], "id": cid, "doc_slice_radius": k})
    return out, dict(used)


def simulate_sources(name: str, policy: Dict[str, int], sources, k0_mode: str = "skip", questions=None, verbose=False):
    recs, used = assemble_from_sources(policy, sources, k0_mode)
    table = build_table(name, recs)
    res = evaluate(table, questions, verbose)
    res["k_dist"] = dict(sorted(Counter(policy.values()).items()))
    res["sources_used"] = used
    return res


def save(res, path):
    json.dump({"avg": res["avg"], "per_q": res["per_q"], "top20": res["top20"], "k_dist": res.get("k_dist")}, open(path, "w"))
