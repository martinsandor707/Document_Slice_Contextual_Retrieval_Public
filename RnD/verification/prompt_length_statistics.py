"""Length statistics, in characters and in nomic tokens, of every prompt the pipeline sends to the language model:
  * the summariser SYSTEM prompt (RnD/Models/chunker_full_doc.Modelfile; the baseline's chunker_anthropic.Modelfile
    is compared with it) and the two summariser USER messages ("FULL DOCUMENT:" + slice, "CHUNK:" + chunk) per
    call, for the full-document baseline, the fixed radii k = 0..4 and the routed pipeline (primary and replicate);
  * the router SYSTEM prompt (BINARY_SYSTEM_PROMPT) and the router USER prompt (ten exemplars + clipped query) for
    every chunk that reaches Tier 2 (the 42 table-gated chunks never see the router prompt).
Messages are rebuilt exactly as RnD/ablation_dynamic_doc_slice_generation.ipynb (slice pipelines),
RnD/anthropic_traditional_chunking.ipynb (baseline; its "FULL DOCUMENT:" header is repeated j+1 times for the
j-th chunk of a document, reported as "as sent" next to the nominal single header) and
utils.dynamic_slice_prediction.DynamicSlicePredictor.build_messages (router) construct them.
Tokens are counted with the nomic-embed-text-v1.5 tokenizer (tokenizer.tokenize, no special tokens), the same
tokenizer used for every token figure of the manuscript; it is a proxy for the Gemma tokenizer of the served model
(the router notebook's Ollama-measured prompt_eval_count, mean 2 062, is printed next to the proxy for calibration).
Backs: Sect. 3.2 (system prompt 789 characters / ~190 tokens, prompt bound s+(2k+2)*2000), Sect. 3.3 (router prompt
~2 000 tokens), Sect. 3.5 / Fig. 4 (prompt budgets incl. system prompts and headers), Table 3.
Run from the repository root:  .venv/bin/python RnD/verification/prompt_length_statistics.py   (CPU only, ~1 min)"""
import json, re, statistics, sys
from _common import RND, SPLIT, DYNAMIC_JSON, REAL_RUNS, load_docs, nomic_tokenizer

sys.path.insert(0, str(RND))
import transformers; transformers.logging.set_verbosity_error()      # silence the >8192-token length warnings
from utils.dynamic_slice_prediction import DynamicSlicePredictor, table_evidence

tok = nomic_tokenizer()
ntok = lambda s: len(tok.tokenize(s))
NUM_CTX_SLICE, NUM_CTX_BASELINE = 16384, 30000

# ---------------------------------------------------------------------------------------------- summariser prompts
def load_modelfile_system(path):
    """Same extraction as load_modelfile() in the dynamic notebook: the SYSTEM block, kept verbatim (incl. newlines)."""
    text = open(path, encoding="utf-8").read()
    m = re.search(r'^SYSTEM\s+"""(.*?)"""', text, flags=re.S | re.M)
    ctx = re.search(r"^PARAMETER\s+num_ctx\s+(\d+)", text, flags=re.M)
    return m.group(1), int(ctx.group(1))

sys_full, ctx_full = load_modelfile_system(RND / "Models" / "chunker_full_doc.Modelfile")
sys_anth, ctx_anth = load_modelfile_system(RND / "Models" / "chunker_anthropic.Modelfile")
print("== summariser SYSTEM prompt (Models/chunker_full_doc.Modelfile, slice and routed pipelines)")
print(f"   {len(sys_full)} characters, {ntok(sys_full)} nomic tokens; num_ctx {ctx_full}")
print(f"   baseline Modelfile (chunker_anthropic): {'identical text' if sys_anth == sys_full else 'DIFFERENT text'}, num_ctx {ctx_anth}")

docs = load_docs(SPLIT)
chunk_tok = {name: [ntok(c["text"]) for c in ch] for name, ch in docs.items()}

def window(n, i, k):
    """Indices of the boundary-shifted window exactly as the generation notebooks build it (min(2k+1, n) chunks)."""
    lo, hi = i - k, i + k
    lo_t, hi_t = max(0, lo), min(n - 1, hi)
    if lo < 0: hi_t = min(n - 1, hi_t + abs(lo))
    if hi > n - 1: lo_t = max(0, lo_t - abs(hi - hi_t))
    return range(lo_t, hi_t + 1)

def routing(records):
    out, counters = {}, {}
    for r in records:
        j = counters.get(r["document"], 0); counters[r["document"]] = j + 1
        out[(r["document"], j)] = r["doc_slice_radius"]
    return out
primary = routing(json.load(open(DYNAMIC_JSON, encoding="utf-8")))
replicate = routing(json.load(open(REAL_RUNS / "binary_t2_heldout_chunks.json", encoding="utf-8")))

def summariser_calls(policy, as_sent_header=False):
    """Yield (doc, i, msg1, msg2) for every summariser call of a policy."""
    for name, chunks in docs.items():
        texts = [c["text"] for c in chunks]
        for i, text in enumerate(texts):
            k = policy(name, i)
            if k == "full":
                body = " ".join(texts)                                  # anthropic_traditional_chunking.ipynb: entire_doc = " ".join(chunks_str)
                msg1 = "FULL DOCUMENT:\n" * ((i + 1) if as_sent_header else 1) + body
            elif k == "self":                                           # paper's fixed k=0 ablation: the slice is the chunk itself
                msg1 = "FULL DOCUMENT:\n" + " " + text
            elif k > 0:
                msg1 = "FULL DOCUMENT:\n" + "".join(" " + texts[j] for j in window(len(texts), i, k))
            else:
                continue                                                # routed k = 0: no summariser call
            yield name, i, msg1, f"CHUNK:\n{text}"

def stats(values):
    return f"min {min(values):>7.0f}  median {statistics.median(values):>8.0f}  mean {statistics.mean(values):>8.0f}  max {max(values):>7.0f}  sum {sum(values):>11.0f}"

policies = {"full document (nominal, one header)": (lambda n, i: "full", False),
            "full document (as sent, repeated header)": (lambda n, i: "full", True),
            "fixed k=0 (self-slice)": (lambda n, i: "self", False)}
for kk in range(1, 5):
    policies[f"fixed k={kk}"] = ((lambda kk: (lambda n, i: kk))(kk), False)
policies["routed pipeline (primary run)"] = (lambda n, i: primary[(n, i)], False)
policies["routed pipeline (replicate)"] = (lambda n, i: replicate[(n, i)], False)

print("\n== summariser USER messages per call (message 1 = 'FULL DOCUMENT:' + slice; message 2 = 'CHUNK:' + chunk); "
      "'total' = system + message 1 + message 2")
sys_c, sys_t = len(sys_full), ntok(sys_full)
for pname, (policy, as_sent) in policies.items():
    calls = list(summariser_calls(policy, as_sent))
    c1 = [len(m1) for _, _, m1, _ in calls]; c2 = [len(m2) for _, _, _, m2 in calls]
    t1 = [ntok(m1) for _, _, m1, _ in calls]; t2 = [ntok(m2) for _, _, _, m2 in calls]
    tot_c = [sys_c + a + b for a, b in zip(c1, c2)]; tot_t = [sys_t + a + b for a, b in zip(t1, t2)]
    over16 = sum(t > NUM_CTX_SLICE for t in tot_t); over30 = sum(t > NUM_CTX_BASELINE for t in tot_t)
    print(f"-- {pname}: {len(calls)} calls")
    print(f"   message 1 chars   {stats(c1)}")
    print(f"   message 1 tokens  {stats(t1)}")
    print(f"   message 2 chars   {stats(c2)}")
    print(f"   message 2 tokens  {stats(t2)}")
    print(f"   total chars       {stats(tot_c)}")
    print(f"   total tokens      {stats(tot_t)}\n   calls above {NUM_CTX_SLICE} tokens: {over16}, above {NUM_CTX_BASELINE}: {over30} (nomic-token proxy)")
    if pname.startswith("routed pipeline (primary"):
        content = sum(len(m1) - len("FULL DOCUMENT:\n") - m1.count("\n") + 0 for _, _, m1, _ in calls)  # placeholder, replaced below
        slice_plus_chunk = sum((len(m1) - len("FULL DOCUMENT:\n") - len(list(window(len(docs[n]), i, primary[(n, i)])))) + (len(m2) - len("CHUNK:\n")) for n, i, m1, m2 in calls)
        print(f"   cross-check: slice + chunk characters without headers/spaces = {slice_plus_chunk} (prompt_volume_and_attention_proxy.py reports 4 391 194)")
print(f"\n   bound of one slice prompt, nomic tokens: system {sys_t} + (2k+2)*2000 -> k=2: {sys_t + 12000}, k=3: {sys_t + 16000} (num_ctx {NUM_CTX_SLICE})")

# ---------------------------------------------------------------------------------------------- router prompts
pred = DynamicSlicePredictor()               # module defaults; build_messages() never contacts the Ollama server
sample_msgs = pred.build_messages("x")
sys_r = sample_msgs[0]["content"]
prefix = sample_msgs[1]["content"][: sample_msgs[1]["content"].rfind("Passage:\n")]     # ten exemplar blocks
print("\n== router SYSTEM prompt (BINARY_SYSTEM_PROMPT)")
print(f"   {len(sys_r)} characters, {ntok(sys_r)} nomic tokens")
print(f"== router USER prompt: exemplar prefix (10 held-out exemplars) {len(prefix)} characters, {ntok(prefix)} nomic tokens; "
      f"query = 'Passage:' + clip(chunk, 1500 head + 500 tail) + 'Answer:'")
tier2 = [(name, i, c["text"]) for name, ch in docs.items() for i, c in enumerate(ch) if table_evidence(c["text"]) is None]
uc, ut, clipped = [], [], 0
for name, i, text in tier2:
    user = pred.build_messages(text)[1]["content"]
    uc.append(len(user)); ut.append(ntok(user))
    clipped += len(text.strip()) > pred.query_head + pred.query_tail + 20
print(f"   chunks reaching Tier 2: {len(tier2)} of {sum(len(ch) for ch in docs.values())} ({sum(len(ch) for ch in docs.values()) - len(tier2)} table-gated); chunks clipped: {clipped}")
print(f"   user prompt chars   {stats(uc)}")
print(f"   user prompt tokens  {stats(ut)}")
tot = [ntok(sys_r) + t for t in ut]
print(f"   system + user tokens {stats(tot)}")
print(f"   calibration: the router notebook (RnD/ollama_dynamic_slice_length.ipynb) recorded an Ollama prompt_eval_count mean of 2 062 and max 2 602 "
      f"for the 4-class prompt; nomic-proxy mean here {statistics.mean(tot):.0f} -> Gemma tokens per nomic token ~ {2062/statistics.mean(tot):.2f} (different prompt wording, indicative only)")
print(f"   total router prompt volume over the corpus: {sum(uc) + len(tier2) * len(sys_r):,} characters, {sum(tot):,} nomic tokens "
      f"(the exemplar prefix is identical in every call and is served from the model's cache)")
