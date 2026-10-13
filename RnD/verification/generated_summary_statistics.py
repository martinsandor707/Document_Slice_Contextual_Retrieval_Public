"""Descriptive statistics (min / Q1 / median / mean / Q3 / max, plus s.d.) of the generated context summaries, in
characters and in nomic tokens, for every stored generation: the full-document baseline, the conference paper's
k=3 run, the fixed radii k = 0..4 of the ablation, the routed runs (primary notebook run, replicate, binary_t2,
tables1, constant-k=2 regeneration) and the policy corpus (k=3 and full-document). Summaries are the 'context'
field of the stored chunk files; an empty context means the chunk was embedded without a summariser call (routed
k = 0). Extra column per set: summaries that open with the forbidden phrase 'This chunk'. Two further checks run
silently (SHOW_EXAMPLE_COPIES turns the first one on; the second prints only on a mismatch): summaries that are a verbatim copy of one of the three example
outputs of the summariser's system prompt (the degeneration the router study found for wide slices), and the
consistency of the embedded text (text == context + blank line + original chunk).
Backs Sect. 3.2 ('typically one or two sentences, about 200 characters'), Sect. 4.3 (summary length of the routed
run, 'degenerate table summaries') and the noise-floor discussion of Sect. 4.3 in sn_green_rag_article.tex.
Tokens: nomic-embed-text-v1.5 tokenizer (tokenizer.tokenize, no special tokens), as for every token figure.
Run from the repository root:  .venv/bin/python RnD/verification/generated_summary_statistics.py   (CPU only)"""
import json, re, statistics, collections
from _common import RND, REAL_RUNS, DYNAMIC_JSON, nomic_tokenizer

PRE = RND / "preprocessed_chunks"
SETS = [
    ("full-document baseline (anthropic_control_table)", PRE / "anthropic_control_chunks_with_metadata.json"),
    ("conference paper k=3 run (my_anthropic_sliding_table)", PRE / "anthropic_sliding_chunks_with_metadata.json"),
    ("fixed k=0 (chunk-only summary)", PRE / "ablation_doc_slice_radius_0.json"),
    ("fixed k=1", PRE / "ablation_doc_slice_radius_1.json"),
    ("fixed k=2", PRE / "ablation_doc_slice_radius_2.json"),
    ("fixed k=3", PRE / "ablation_doc_slice_radius_3.json"),
    ("fixed k=4", PRE / "ablation_doc_slice_radius_4.json"),
    ("constant k=2 through the routed notebook (fixed2)", REAL_RUNS / "fixed2_chunks.json"),
    ("4-class router, tables->1 (tables1)", REAL_RUNS / "tables1_chunks.json"),
    ("2-class router, benchmark exemplars (binary_t2)", REAL_RUNS / "binary_t2_chunks.json"),
    ("routed pipeline, replicate (binary_t2_heldout)", REAL_RUNS / "binary_t2_heldout_chunks.json"),
    ("routed pipeline, primary (notebook run)", DYNAMIC_JSON),
    ("policy corpus, k=3 (tobacco_sliding)", PRE / "tobacco_sliding.json"),
    ("policy corpus, full document (tobacco_traditional)", PRE / "tobacco_traditional.json"),
]
EXAMPLES = {"Introduces the central thesis about climate policy reform.",
            "Provides supporting evidence for the argument on economic inequality.",
            "Transitions from background information to proposed methodology."}
tok = nomic_tokenizer()

def six(values):
    """min / median / mean / max and s.d. (Q1 and Q3 are computed but left out of the console table for readability;
    set SHOW_QUARTILES = True to print them)."""
    q = statistics.quantiles(values, n=4, method="inclusive")
    if SHOW_QUARTILES:
        return f"{min(values):>5.0f} {q[0]:>6.0f} {statistics.median(values):>7.0f} {statistics.mean(values):>7.1f} {q[2]:>6.0f} {max(values):>5.0f}  sd {statistics.pstdev(values):>5.1f}"
    return f"{min(values):>5.0f} {statistics.median(values):>7.0f} {statistics.mean(values):>7.1f} {max(values):>5.0f} {statistics.pstdev(values):>6.1f}"

SHOW_QUARTILES = False
SHOW_EXAMPLE_COPIES = False      # verbatim copies of a system-prompt example: computed, printed only when True

if SHOW_QUARTILES:
    print(f"{'set':54s} {'recs':>4s} {'summ':>4s} | {'chars: min':>10s} {'Q1':>6s} {'median':>7s} {'mean':>7s} {'Q3':>6s} {'max':>5s} {'':>8s} | {'tokens: min':>11s} {'Q1':>6s} {'median':>7s} {'mean':>7s} {'Q3':>6s} {'max':>5s} {'':>8s} | 'This chunk'")
else:
    print(f"{'set':54s} {'recs':>4s} {'summ':>4s} | {'chars: min':>10s} {'median':>7s} {'mean':>7s} {'max':>5s} {'sd':>6s} | {'tokens: min':>11s} {'median':>7s} {'mean':>7s} {'max':>5s} {'sd':>6s} | 'This chunk'")
per_k = {}
for label, path in SETS:
    recs = json.load(open(path, encoding="utf-8"))
    summ = [r["context"] for r in recs if r.get("context")]
    chars = [len(s) for s in summ]; toks = [len(tok.tokenize(s)) for s in summ]
    this_chunk = sum(bool(re.match(r"\s*this chunk", s, flags=re.I)) for s in summ)
    copies = sum(s.strip() in EXAMPLES for s in summ)
    ok = all(r["text"] == (r["context"] + "\n\n" + r["original_text"] if r.get("context") else r["original_text"]) for r in recs)
    print(f"{label:54s} {len(recs):4d} {len(summ):4d} | {six(chars)} | {six(toks)} | {this_chunk:11d}")
    if copies and SHOW_EXAMPLE_COPIES:
        print(f"      note: {copies} summaries are verbatim copies of a system-prompt example sentence")
    if not ok:
        print("      WARNING: 'text' field is not context + blank line + original chunk for every record")
    if "doc_slice_radius" in recs[0]:
        per_k[label] = collections.defaultdict(list)
        for r in recs:
            if r.get("context"):
                per_k[label][(r["doc_slice_radius"], r.get("slice_tier", "?"))].append(len(r["context"]))

print("\n== breakdown of the routed runs by (radius, tier), summary characters")
for label, groups in per_k.items():
    for key, vals in sorted(groups.items()):
        print(f"   {label:54s} k={key[0]} tier={key[1]:5s} n={len(vals):3d} | {six(vals)}")
print("\ns.d. = population standard deviation; 'summ' = non-empty summaries (summariser calls); Q1/Q3 (statistics.quantiles, "
      "inclusive method) are available with SHOW_QUARTILES = True; chars per token of the summaries = mean chars / mean tokens.")
